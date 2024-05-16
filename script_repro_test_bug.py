from contextlib import nullcontext
import dataclasses
import os
import pathlib
import tempfile

from composer.utils import dist
from tests.trainer.test_checkpoint import TestCheckpointResumption, _assert_checkpoints_equivalent
from tests.trainer.test_fsdp_checkpoint import FSDPConfig, get_trainer


def test_fsdp_monolith_resumption(
    device: str,
    world_size: int,
    use_orig_params: bool,
    sync_module_states: bool,
    tmp_path: pathlib.Path,
    model_1_init_device: str,
    model_2_init_device: str,
):
    save_interval = '1ba'
    save_filename = 'ba{batch}-rank{rank}.pt'
    resume_file = 'ba1-rank{rank}.pt'
    final_checkpoint = 'latest-rank{rank}.pt'
    fsdp_config = FSDPConfig(
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
        state_dict_type='full',
    )

    # All ranks use rank 0 folder
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = pathlib.Path(tmp_paths[0])

    trainer_1 = get_trainer(
        save_folder=os.path.join(save_folder, 'first'),
        save_filename=save_filename,
        save_interval=save_interval,
        fsdp_config=fsdp_config,
        precision='amp_fp16',
        max_duration='1ep',
    )

    trainer_1.fit()
    trainer_1.close()

    TestCheckpointResumption._assert_expected_num_checkpoints(
        save_folder=os.path.join(save_folder, 'first'),
        save_interval=save_interval,
        num_epochs=1,  # set in get_trainer()
        num_batches_per_epoch=8,  # set in get_trainer()
        is_deepspeed=False,
    )

    resume_file = os.path.join(save_folder, 'first', resume_file)
    model_init_device = [model_1_init_device, model_2_init_device][dist.get_global_rank()]
    fsdp_config_dict = dataclasses.asdict(fsdp_config)
    fsdp_config_dict['load_monolith_rank0_only'] = True
    fsdp_config = FSDPConfig(**fsdp_config_dict)

    success = (sync_module_states == True and model_1_init_device == 'cpu')

    with nullcontext():
        trainer_2 = get_trainer(
            model_init_device=model_init_device,
            save_folder=os.path.join(save_folder, 'second'),
            save_filename=save_filename,
            save_interval=save_interval,
            fsdp_config=fsdp_config,
            precision='amp_fp16',
            max_duration='1ep',
            load_path=resume_file,  # <-- resume training from file
        )
        trainer_2.fit()
        trainer_2.close()

        _assert_checkpoints_equivalent(
            save_folder / 'first' / final_checkpoint,
            save_folder / 'second' / final_checkpoint,
        )

if __name__ == 'main':
    # tmp path context manager
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_fsdp_monolith_resumption(device='gpu', world_size=2, use_orig_params=False, sync_module_states=True, tmp_path=tmp_path, model_1_init_device='cpu', model_2_init_device='meta')