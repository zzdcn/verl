# VERL 推理引擎与训练引擎架构文档

本文档详细说明 verl 框架的推理引擎（Rollout Engine）和训练引擎（Training Engine）的实现原理及底层调用逻辑。

---

## 1. 架构概览

VERL 采用 **单控制器（Single Controller）+ 分布式 Worker** 的架构模式。主控进程（Trainer）运行在单个 CPU/GPU 节点上，通过 Ray 调度多个 Worker 完成分布式训练和推理任务。

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          VERL 整体架构                                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    Single Controller (Driver)                      │   │
│  │                                                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ RayPPOTrainer│  │  DataLoader  │  │  CheckpointEngineManager │  │   │
│  │  └──────┬───────┘  └──────────────┘  └──────────────────────────┘  │   │
│  │         │                                                          │   │
│  │         │ Ray Remote Calls                                         │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    RayWorkerGroup                            │  │   │
│  │  │  - dispatch_fn: 数据分发                                      │  │   │
│  │  │  - collect_fn: 结果收集                                       │  │   │
│  │  │  - execute_fn: 远程执行                                       │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    Distributed Workers (Ray Actors)                │   │
│  │                                                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │           ActorRolloutRefWorker (Hybrid Engine)               │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│ │   │
│  │  │  │   Actor     │  │  Reference  │  │       Rollout           ││ │   │
│  │  │  │(FSDP/Engine)│  │   Policy    │  │  (SGLang/vLLM/TRT-LLM)  ││ │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘│ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                    │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐│   │
│  │  │  Critic Worker │  │  Reward Model  │  │   AgentLoopWorker      ││   │
│  │  │  (FSDP/Engine) │  │    Worker      │  │  (Tool Orchestration)  ││   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘│   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 训练引擎（Training Engine）

### 2.1 BaseEngine 接口

所有训练引擎都继承自 `BaseEngine` 抽象基类，定义了统一的训练接口。

**源码位置**: `verl/workers/engine/base.py:29-337`

```python
class BaseEngine:
    """训练引擎抽象基类"""

    def initialize(self):
        """初始化模型、优化器、学习率调度器"""

    def train_mode(self, **kwargs) -> ContextManager:
        """进入训练模式的上下文管理器"""

    def eval_mode(self, **kwargs) -> ContextManager:
        """进入评估模式的上下文管理器"""

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """执行一个训练 step"""
        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        return outputs

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
        """执行推理（不更新梯度）"""
        with torch.no_grad():
            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
        return outputs

    def get_per_tensor_param(self) -> tuple[Generator, Optional[dict]]:
        """获取逐层参数生成器，用于权重同步"""

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, **kwargs):
        """保存检查点"""

    def load_checkpoint(self, local_path, hdfs_path=None, **kwargs):
        """加载检查点"""
```

### 2.2 Engine 注册机制

VERL 使用注册机制支持多种训练后端。

**源码位置**: `verl/workers/engine/base.py:266-337`

```python
class EngineRegistry:
    """训练引擎注册中心"""
    _engines = {}

    @classmethod
    def register(cls, model_type: str, backend: list[str] | str, device: list[str] | str = "cuda"):
        """装饰器：注册引擎类"""
        def decorator(engine_class):
            # 按 model_type -> backend -> device 三级索引存储
            cls._engines[model_type][backend][device] = engine_class
            return engine_class
        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        """根据 model_type 和 backend 获取引擎类"""
        device = get_device_name()  # 自动检测设备类型
        return cls._engines[model_type][backend][device]
```

**支持的引擎后端**:
- `fsdp`: PyTorch FSDP (Fully Sharded Data Parallel)
- `megatron`: Megatron-LM 张量/流水线并行

### 2.3 TrainingWorker

`TrainingWorker` 是对 `BaseEngine` 的高级封装，提供 Ray 远程调用接口。

**源码位置**: `verl/workers/engine_workers.py:53-369`

```python
class TrainingWorker(Worker, DistProfilerExtension):
    """训练 Worker，封装 BaseEngine 提供 Ray 接口"""

    def __init__(self, config: TrainingWorkerConfig):
        # 1. 初始化分布式进程组
        initialize_global_process_group_ray(timeout_second=None)

        # 2. 创建训练引擎
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,      # "language_model" 或 "value_model"
            backend=self.engine_config.strategy,    # "fsdp" 或 "megatron"
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
        )

        # 3. 注册数据分发/收集信息
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_batch(self, data: TensorDict) -> TensorDict:
        """训练一个 batch"""
        with self.engine.train_mode():
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
        return self._postprocess_output(output)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def infer_batch(self, data: TensorDict) -> TensorDict:
        """推理一个 batch（用于计算 log prob / value）"""
        with self.engine.eval_mode():
            output = self.engine.infer_batch(data, loss_function=self.loss_fn)
        return self._postprocess_output(output)
```

### 2.4 FSDP Workers 实现

FSDP 是 verl 的默认训练后端，支持大模型的分布式训练。

**源码位置**: `verl/workers/fsdp_workers.py:140-300`

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """混合 Worker：集成 Actor、Rollout、Reference 三个角色"""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        # 1. 初始化分布式进程组
        torch.distributed.init_process_group(
            backend=f"cpu:gloo,{device_name}:{nccl_backend}",
            rank=rank,
            world_size=world_size,
        )

        # 2. 创建 FSDP Device Mesh
        self.device_mesh = create_device_mesh(world_size, fsdp_size)

        # 3. 创建 Ulysses Sequence Parallel Device Mesh（可选）
        if ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(dp, ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"]
            )
```

**Device Mesh 创建逻辑**:

```python
def create_device_mesh(world_size, fsdp_size):
    """创建 FSDP Device Mesh"""
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 纯 FSDP 模式
        device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(world_size,),
            mesh_dim_names=["fsdp"]
        )
    else:
        # HSDP (Hybrid Sharded Data Parallel) 模式
        device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh
```

---

## 3. 推理引擎（Rollout Engine）

### 3.1 BaseRollout 接口

**源码位置**: `verl/workers/rollout/base.py:29-103`

```python
class BaseRollout(ABC):
    """Rollout 引擎基类"""

    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        self.config = omega_conf_to_dataclass(config)
        self.model_config = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """恢复 GPU 显存占用（权重/KV Cache）"""
        pass

    @abstractmethod
    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """从训练引擎同步权重"""
        pass

    @abstractmethod
    async def release(self):
        """释放 GPU 显存"""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """同步模式批量生成（已废弃，推荐使用异步模式）"""
        raise NotImplementedError
```

### 3.2 Rollout 引擎注册

**源码位置**: `verl/workers/rollout/base.py:81-103`

```python
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}

def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    """根据名称获取 Rollout 类"""
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
```

### 3.3 SGLang Rollout 实现

SGLang 是 verl 的默认推理后端，提供高性能的 LLM 推理服务。

**源码位置**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:88-217`

```python
class ServerAdapter(BaseRollout):
    """SGLang 服务器适配器，用于 Hybrid 模式下的权重同步"""

    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        # 计算当前进程的位置
        rank = int(os.environ["RANK"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size

    async def _init_server_adapter(self):
        """延迟初始化 HTTP Server Adapter"""
        if self._engine is not None:
            return

        # 只有 TP rank 0 需要初始化 HTTP adapter
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # 获取 SGLang Server 的地址
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()

        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=server_address,
            port=server_port,
            launch_server=False,  # Server 已由外部启动
        )

    async def resume(self, tags: list[str]):
        """恢复 GPU 显存占用"""
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """释放 GPU 显存"""
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """从训练引擎同步权重到 SGLang Server"""
        await self._init_server_adapter()

        bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20

        # 分桶传输权重
        async for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
            await sgl_update_weights(
                engine=self._engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        # 清空推理缓存
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
```

### 3.4 Rollout 模式

VERL 支持三种 Rollout 模式：

**源码位置**: `verl/workers/rollout/replica.py:45-58`

```python
class RolloutMode(Enum):
    """Rollout 运行模式"""

    HYBRID = "hybrid"
    """
    Hybrid 模式：Rollout 和 Training 共享 GPU
    - 训练时：Training Engine 占用 GPU
    - 推理时：Rollout Engine 占用 GPU
    - 需要权重同步和显存切换
    """

    COLOCATED = "colocated"
    """
    Colocated 模式：Rollout 和 Training 在同一 Placement Group
    - 不同进程，但在相同节点
    - 可以通过 CUDA IPC 高效共享数据
    """

    STANDALONE = "standalone"
    """
    Standalone 模式：Rollout 和 Training 使用独立 GPU
    - 完全解耦的架构
    - 适用于大规模分布式训练
    """
```

---

## 4. ActorRolloutRefWorker（混合引擎）

`ActorRolloutRefWorker` 是 verl 的核心 Worker 类，集成了 Actor、Rollout、Reference 三个角色。

**源码位置**: `verl/workers/engine_workers.py:371-637`

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """混合 Worker：包含 Actor 模型、Rollout 引擎和可选的 Reference 模型"""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        self.config = config
        self.role = role  # "actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"

        self._is_actor = "actor" in self.role
        self._is_rollout = "rollout" in self.role
        self._is_ref = "ref" in self.role

        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """初始化所有模型组件"""

        # 1. 初始化 Reference 模型（可选）
        if "ref" in self.role:
            ref_training_config = TrainingWorkerConfig(...)
            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()

        # 2. 初始化 Actor 模型
        if "actor" in self.role:
            actor_training_config = TrainingWorkerConfig(...)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)

        # 3. 初始化 Rollout 引擎
        if "rollout" in self.role:
            rollout_config = omega_conf_to_dataclass(self.config.rollout)

            # 创建 Rollout Device Mesh
            rollout_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(dp, infer_tp, infer_pp),
                mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 获取并实例化 Rollout 类
            rollout_cls = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config,
                model_config=model_config,
                device_mesh=rollout_device_mesh
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """计算 Actor 的 log probability"""
        output = self.actor.infer_batch(data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        """计算 Reference 的 log probability"""
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: TensorDict) -> TensorDict:
        """执行 Actor PPO 更新"""
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """将训练权重同步到 Rollout 引擎"""

        # 1. 恢复 Rollout 的权重显存
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])

        # 2. 从 Actor 获取权重生成器
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()

        # 3. 更新 Rollout 权重
        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)

        # 4. 将 Actor 模型卸载到 CPU（释放 GPU 显存）
        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 5. 恢复 KV Cache 显存
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
```

---

## 5. Ray 分布式调度

### 5.1 ResourcePool 管理

**源码位置**: `verl/single_controller/ray/base.py:98-315`

```python
class RayResourcePool(ResourcePool):
    """Ray 资源池：管理 Placement Group"""

    def __init__(self, process_on_nodes: list[int], use_gpu: bool = True, max_colocate_count: int = 10):
        """
        Args:
            process_on_nodes: 每个节点的进程数，例如 [8, 8] 表示 2 个节点各 8 GPU
            use_gpu: 是否使用 GPU
            max_colocate_count: 每个 GPU 上最多同时放置的 Actor 数
        """
        self._store = process_on_nodes
        self.use_gpu = use_gpu
        self.max_colocate_count = max_colocate_count

    def get_placement_groups(self, strategy="STRICT_PACK") -> list[PlacementGroup]:
        """创建 Ray Placement Groups"""
        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle["GPU"] = 1

        # 每个节点一个 Placement Group
        pg_scheme = [[bundle.copy() for _ in range(process_count)]
                     for process_count in self._store]

        pgs = [placement_group(bundles=bundles, strategy=strategy) for bundles in pg_scheme]
        ray.get([pg.ready() for pg in pgs])

        return sort_placement_group_by_node_ip(pgs)
```

### 5.2 RayWorkerGroup

**源码位置**: `verl/single_controller/ray/base.py:397-882`

```python
class RayWorkerGroup(WorkerGroup):
    """Ray Worker 组：管理一组 Ray Actors"""

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        self.resource_pool = resource_pool
        self.ray_cls_with_init = ray_cls_with_init
        self._workers = []

        # 创建 Workers
        pgs = resource_pool.get_placement_groups()
        for pg_idx, pg in enumerate(pgs):
            for local_rank in range(local_world_size):
                worker = self._create_worker(rank, pg, local_rank, ...)
                self._workers.append(worker)

        # 绑定方法
        self._bind_worker_method(ray_cls_with_init.cls, func_generator)

    def _create_worker(self, rank, pg, local_rank, resource_pool, ray_cls_with_init, **kwargs):
        """创建单个 Worker Actor"""
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }

        ray_cls_with_init.update_options({
            "runtime_env": {"env_vars": env_vars},
            "name": f"{self.name_prefix}_{pg_idx}:{local_rank}",
        })

        # 使用 Placement Group 调度策略
        worker = ray_cls_with_init(
            placement_group=pg,
            placement_group_bundle_idx=local_rank,
            use_gpu=True,
            num_gpus=1 / resource_pool.max_colocate_count,
        )
        return worker

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """异步执行：向所有 Workers 发送任务"""
        return [worker.method_name.remote(*args, **kwargs) for worker in self._workers]

    def spawn(self, prefix_set):
        """拆分 Worker Group：为不同角色创建子 Group"""
        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_wg = self.from_detached(worker_handles=self._workers, ...)
            _rebind_actor_methods(new_wg, prefix)
            new_worker_group_dict[prefix] = new_wg
        return new_worker_group_dict
```

### 5.3 Dispatch 装饰器

VERL 使用装饰器模式实现数据分发和结果收集。

**源码位置**: `verl/single_controller/base/decorator.py`

```python
class Dispatch(Enum):
    """数据分发模式"""

    ONE_TO_ALL = "one_to_all"
    """广播模式：相同数据发送到所有 Workers"""

    DP_COMPUTE = "dp_compute"
    """数据并行模式：数据按 DP 维度切分，结果合并"""

    DP_COMPUTE_PROTO = "dp_compute_proto"
    """DataProto 数据并行模式：支持 DataProto 的分发和收集"""

@register(dispatch_mode=Dispatch.DP_COMPUTE)
def train_batch(self, data):
    """
    dispatch_fn: 将 data 按 dp_rank 切分，发送给对应 Worker
    execute_fn: Worker 执行 train_batch
    collect_fn: 收集所有 Worker 的结果并合并
    """
    pass
```

---

## 6. Agent Loop（多轮推理）

### 6.1 AgentLoopManager

**源码位置**: `verl/experimental/agent_loop/agent_loop.py:845-1023`

```python
class AgentLoopManager:
    """Agent Loop 管理器：协调多个 AgentLoopWorker"""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, rollout_resource_pool: RayResourcePool):
        # 1. 初始化 LLM Server
        self.rollout_replicas = self._init_llm_servers(rollout_resource_pool)

        # 2. 创建 AgentLoopWorker
        self.agent_loop_workers = self._create_workers(config, self.rollout_replicas)

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """批量生成：将 batch 分发给 Workers 并行执行"""
        # 1. 按 Worker 数量切分 batch
        batches = batch.chunk(num_workers)

        # 2. 并行执行
        futures = [worker.generate_sequences.remote(b) for worker, b in zip(self.agent_loop_workers, batches)]

        # 3. 收集结果
        outputs = ray.get(futures)
        return DataProto.concat(outputs)
```

### 6.2 AgentLoopWorker

**源码位置**: `verl/experimental/agent_loop/agent_loop.py:345-806`

```python
class AgentLoopWorker:
    """Agent Loop Worker：处理多轮对话和工具调用"""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        # LLM Server 负载均衡器
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        # Tokenizer 和 Processor
        self.tokenizer = hf_tokenizer(model_path)
        self.processor = hf_processor(model_path)

        # Reward Loop Worker（可选）
        if config.reward_model.use_reward_loop:
            self.reward_loop_worker = RewardLoopWorker.remote(config)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """批量多轮推理"""
        # 1. 为每个样本创建 AgentLoop 实例
        agent_loops = [
            self._create_agent_loop(sample) for sample in batch
        ]

        # 2. 并发执行所有 Agent Loop
        tasks = [loop.run(sampling_params=..., **sample.to_dict()) for loop, sample in zip(agent_loops, batch)]
        outputs = await asyncio.gather(*tasks)

        # 3. 转换输出格式
        return self._convert_outputs(outputs)
```

### 6.3 AsyncLLMServerManager

**源码位置**: `verl/experimental/agent_loop/agent_loop.py:57-123`

```python
class AsyncLLMServerManager:
    """LLM Server 负载均衡器"""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        self.server_handles = server_handles

        # 最小请求数负载均衡
        self.weighted_servers = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_servers)

        # LRU 缓存：request_id -> server 映射（支持 Sticky Session）
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        """选择服务器：支持 Sticky Session"""
        # 已有映射则返回相同服务器（多轮对话利用 Prefix Caching）
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # 选择请求数最少的服务器
        _, _, server = self.weighted_servers[0]
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])

        self.request_id_to_server[request_id] = server
        return server

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        """生成 tokens"""
        server = self._choose_server(request_id)
        return await server.generate.remote(
            request_id=uuid4().hex,  # 每轮使用新 request_id
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            **kwargs
        )
```

---

## 7. PPO 训练流程

### 7.1 RayPPOTrainer

**源码位置**: `verl/trainer/ppo/ray_trainer.py:222-400`

```python
class RayPPOTrainer:
    """Ray PPO Trainer：协调分布式 PPO 训练"""

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ...):
        self.tokenizer = tokenizer
        self.config = config

        # Worker 角色映射
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        # KL 控制器
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

    def init_workers(self):
        """初始化所有 Workers"""
        # 1. 创建资源池
        self.resource_pool_manager.create_resource_pool()

        # 2. 创建 Actor-Rollout Worker
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout_ref",
        )

        # 3. 创建 Colocated Worker（多角色共享 GPU）
        worker_dict_cls = create_colocated_worker_cls(class_dict={
            "actor_rollout_ref": actor_rollout_cls,
            "critic": critic_cls,
        })

        # 4. 创建 Worker Group
        wg_dict = RayWorkerGroup(resource_pool=actor_rollout_resource_pool, ray_cls_with_init=worker_dict_cls)
        all_wg = wg_dict.spawn(prefix_set=class_dict.keys())

        # 5. 初始化各角色 Worker
        self.actor_rollout_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_wg.init_model()

        self.critic_wg = all_wg["critic"]
        self.critic_wg.init_model()

        # 6. 创建 AgentLoopManager
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
```

### 7.2 训练主循环

```python
def fit(self):
    """PPO 训练主循环"""
    for epoch in range(total_epochs):
        for batch in self.train_dataloader:
            # 1. Rollout Phase: 生成 responses
            with marked_timer("generate_sequences"):
                gen_batch = self._get_gen_batch(batch)
                output_batch = self.async_rollout_manager.generate_sequences(gen_batch)

            # 2. 计算 Reward
            with marked_timer("compute_reward"):
                reward_tensor, reward_extra_info = compute_reward(batch, self.reward_fn)
                batch.batch["token_level_scores"] = reward_tensor

            # 3. 计算 Reference Log Prob（KL 惩罚）
            if self.use_reference_policy:
                with marked_timer("compute_ref_log_prob"):
                    ref_output = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_output)

            # 4. 计算 Value（Critic）
            if self.use_critic:
                with marked_timer("compute_values"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # 5. 计算 Advantage
            with marked_timer("compute_advantage"):
                batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)

            # 6. Actor PPO 更新
            with marked_timer("update_actor"):
                actor_output = self.actor_rollout_wg.update_actor(batch)

            # 7. Critic 更新
            if self.use_critic:
                with marked_timer("update_critic"):
                    critic_output = self.critic_wg.update_critic(batch)

            # 8. 权重同步：Training -> Rollout
            with marked_timer("update_weights"):
                self.actor_rollout_wg.update_weights()

            # 9. 日志和检查点
            self._log_metrics(actor_output, critic_output)
            if should_save_checkpoint:
                self._save_checkpoint()
```

---

## 8. 权重同步流程

Hybrid 模式下，训练和推理共享 GPU，需要在两者之间切换时同步权重。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         权重同步流程 (Hybrid Mode)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Rollout Phase 结束                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  rollout.release()                                              │    │
│     │  - 释放 KV Cache 显存                                            │    │
│     │  - 释放模型权重显存（如果启用 free_cache_engine）                  │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  2. Training Phase                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  actor.engine.to("cuda")                                        │    │
│     │  - 将 Actor 模型加载到 GPU                                       │    │
│     │  - 执行 forward/backward/optimizer.step                         │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  3. 权重同步                                                                │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  update_weights()                                               │    │
│     │                                                                 │    │
│     │  a. rollout.resume(tags=["weights"])                           │    │
│     │     - 恢复 Rollout 权重显存占用                                  │    │
│     │                                                                 │    │
│     │  b. per_tensor_param = actor.engine.get_per_tensor_param()     │    │
│     │     - 获取 Actor 的逐层权重生成器                                │    │
│     │                                                                 │    │
│     │  c. rollout.update_weights(per_tensor_param)                   │    │
│     │     - 分桶传输权重到 Rollout 引擎                               │    │
│     │     - 使用 CUDA IPC 或 Tensor 直接传输                          │    │
│     │                                                                 │    │
│     │  d. actor.engine.to("cpu")                                     │    │
│     │     - 将 Actor 模型卸载到 CPU，释放 GPU 显存                     │    │
│     │                                                                 │    │
│     │  e. rollout.resume(tags=["kv_cache"])                          │    │
│     │     - 恢复 KV Cache 显存占用                                    │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  4. 新一轮 Rollout Phase                                                    │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  async_rollout_manager.generate_sequences(batch)               │    │
│     │  - 使用更新后的权重进行推理                                      │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键文件索引

| 模块 | 文件路径 | 说明 |
|-----|---------|------|
| **训练引擎** | | |
| BaseEngine | `verl/workers/engine/base.py:29-337` | 训练引擎抽象基类 |
| TrainingWorker | `verl/workers/engine_workers.py:53-369` | 训练 Worker 封装 |
| ActorRolloutRefWorker | `verl/workers/engine_workers.py:371-637` | 混合 Worker（新版） |
| FSDP Workers | `verl/workers/fsdp_workers.py:140-300` | FSDP 实现（旧版） |
| **推理引擎** | | |
| BaseRollout | `verl/workers/rollout/base.py:29-103` | Rollout 引擎基类 |
| SGLang Adapter | `verl/workers/rollout/sglang_rollout/sglang_rollout.py:88-217` | SGLang 适配器 |
| RolloutReplica | `verl/workers/rollout/replica.py:61-340` | Rollout 副本管理 |
| **Ray 调度** | | |
| RayResourcePool | `verl/single_controller/ray/base.py:98-315` | Ray 资源池 |
| RayWorkerGroup | `verl/single_controller/ray/base.py:397-882` | Ray Worker 组 |
| Colocated Worker | `verl/single_controller/ray/base.py:1006-1099` | 共享 GPU Worker |
| **Agent Loop** | | |
| AgentLoopManager | `verl/experimental/agent_loop/agent_loop.py:845-1023` | Agent 管理器 |
| AgentLoopWorker | `verl/experimental/agent_loop/agent_loop.py:345-806` | Agent Worker |
| AsyncLLMServerManager | `verl/experimental/agent_loop/agent_loop.py:57-123` | LLM 负载均衡器 |
| **PPO Trainer** | | |
| RayPPOTrainer | `verl/trainer/ppo/ray_trainer.py:222-400` | PPO 训练器 |
| Core Algorithms | `verl/trainer/ppo/core_algos.py` | PPO/GRPO 核心算法 |
| Loss Functions | `verl/workers/utils/losses.py` | 损失函数 |

---

## 10. 配置示例

### 10.1 Actor-Rollout-Ref 配置

```yaml
actor_rollout_ref:
  hybrid_engine: True  # 启用混合引擎

  model:
    path: Qwen/Qwen2.5-7B-Instruct
    use_remove_padding: True
    lora_rank: 0  # 0 表示不使用 LoRA

  actor:
    fsdp_config:
      strategy: fsdp          # FSDP 策略
      fsdp_size: -1           # -1 表示全 FSDP
      param_offload: False    # 参数卸载
      optimizer_offload: True # 优化器卸载
    ppo_mini_batch_size: 128
    ppo_micro_batch_size_per_gpu: 8

  rollout:
    name: sglang             # 推理引擎
    mode: async              # 异步模式
    tensor_model_parallel_size: 1
    data_parallel_size: 8
    multi_turn:
      enable: True
      max_assistant_turns: 5
    checkpoint_engine:
      backend: naive
      update_weights_bucket_megabytes: 128
```

### 10.2 Critic 配置

```yaml
critic:
  strategy: fsdp
  model_config:
    path: ${actor_rollout_ref.model.path}
  fsdp_config:
    param_offload: False
    optimizer_offload: True
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
```

---

## 11. 常见问题

### Q1: Hybrid 模式和 Standalone 模式如何选择？

- **Hybrid 模式**：推荐用于中小规模训练，GPU 利用率高，但需要处理权重同步开销
- **Standalone 模式**：推荐用于大规模训练，解耦架构更易扩展，但需要更多 GPU

### Q2: 如何切换不同的推理引擎？

修改 `actor_rollout_ref.rollout.name` 配置：
- `sglang`：默认，高性能 LLM 推理
- `vllm`：vLLM 推理引擎
- `trtllm`：TensorRT-LLM 推理引擎

### Q3: 如何调试权重同步问题？

1. 检查 `checkpoint_engine.update_weights_bucket_megabytes` 配置
2. 启用日志：`export VERL_LOGGING_LEVEL=DEBUG`
3. 检查 Device Mesh 配置是否正确

### Q4: Agent Loop 和普通 Rollout 有什么区别？

- **普通 Rollout**：单轮生成，直接调用 LLM 生成 response
- **Agent Loop**：多轮生成，支持工具调用、环境交互、自动重试等

---

## 附录：架构演进

| 版本 | 架构特点 |
|-----|---------|
| v0.x | FSDP Workers 直接实现，紧耦合 |
| v1.x | BaseEngine 抽象，支持多后端 |
| v2.x | Agent Loop 集成，异步推理 |
| 当前 | TrainingWorker + RolloutReplica + AgentLoopManager |

---

*文档生成时间: 2025-01-30*
*VERL 版本: 基于 main 分支*
