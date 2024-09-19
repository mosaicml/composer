

@world_size(4)
@pytest.mark.gpu
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fsdp_trainer_2(world_size: int):
    # from icecream import ic

    ###############
    # Parameters
    ###############

    size: int = 4
    batch_size: int = 1
    num_classes: int = 2
    num_features: int = 2
    seed: int = 44
    tensor_parallel_degree: int = 2
    device: torch.device = torch.device('cuda')
    output_dir: str = '/my-tmp/'

    reproducibility.seed_all(seed)
    rank = dist.get_local_rank()

    ###############
    # DataLoader
    ###############

    my_dataset = MyDataset(
        shape=(num_features,),
        num_classes=num_classes,
        size=size,
        device=device,
        rank=rank,
    )  # X=(num_features,), y=(,), i.e. scalar

    # for i in range(len(my_dataset)):
    #     x, y = my_dataset[i]
    #     ic(rank)
    #     ic(x.shape, x)
    #     ic(y.shape, y)
    #     ic('\n')

    dataloader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(my_dataset),
    )

    # pytorch_dataset = RandomClassificationDataset(
    #     shape=(num_features,),
    #     num_classes=num_classes,
    #     size=size,
    #     device=device,
    # )

    # # clean directory
    # rmtree(output_dir)

    # # columns = {'x': 'ndarray:float32:2', 'y': 'int64'} # 2 -> features
    # columns = {'x': 'pkl', 'y': 'int64'}
    # with MDSWriter(out=output_dir, columns=columns) as out:
    #     for i in range(len(pytorch_dataset)):
    #         x, y = pytorch_dataset[i]
    #         out.write({'x': x.cpu().detach().numpy(), 'y': y.cpu().detach().numpy()})
    #         # out.write({'x': x.numpy(), 'y': y.numpy()})

    # streaming_dataset = StreamingDataset(
    #     local=output_dir,
    #     replication=tensor_parallel_degree,
    #     batch_size=batch_size,
    #     allow_unsafe_types=True
    # )

    # dataloader = DataLoader(
    #     streaming_dataset,
    # )

    ###############
    # Model
    ###############

    model = SimpleComposerMLP(
        num_features=num_features,
        device=device,
        num_classes=num_classes,
    )

    #####################
    # Parallelism Config
    #####################

    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        use_orig_params=True,
    )
    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }
    tp_config = TPConfig(
        layer_plan=layer_plan,
        tensor_parallel_degree=tensor_parallel_degree,
    )
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)

    #####################
    # Trainer
    #####################

    tp_fsdp_trainer = Trainer(
        seed=seed,
        device='gpu',
        model=model,
        max_duration='1ep',
        train_dataloader=dataloader,
        precision='fp32',
        parallelism_config=parallelism_config,
        callbacks=[MemoryMonitor()],
        loggers=[InMemoryLogger()],
        progress_bar=False,
        log_to_console=False,
    )

    tp_fsdp_trainer.fit()
