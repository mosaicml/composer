# ☁️ Train ResNet-50 on AWS

Composer is a PyTorch library that accelerates training for deep learning models while improving quality at significantly lower cost.  Composer makes it possible to train ResNet-50 on the ImageNet dataset to the standard ***76.6% top-1 accuracy*** in ***27 minutes*** on an AWS EC2 instance for a mere ***$15***.  In this tutorial we’ll teach you how simple it is to do this yourself!

The starting point for this tutorial are the training recipes we present in our [Mosaic ResNet blog post](https://www.mosaicml.com/blog/mosaic-resnet).  We’ll walk through:

1. Launching an AWS EC2 instance capable of running GPU training
1. Configuring your AWS EC2 instance to run Composer with our pre-built Docker images
1. Running Composer training using the ResNet-50 Mild recipe introduced in our [blog post](https://www.mosaicml.com/blog/mosaic-resnet)

## Prerequisites

- AWS account with permissions to:
    - Create/manage EC2 instances, EBS volumes
    - Create/manage Security Groups, Key Pairs (alternatively, IT admin provided)
- AWS quota to create [Accelerated Computing](https://aws.amazon.com/ec2/instance-types/?trk=36c6da98-7b20-48fa-8225-4784bced9843&sc_channel=ps&sc_campaign=acquisition&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Compute|EC2|US|EN|Text&s_kwcid=AL!4422!3!536392622533!e!!g!!aws%20instance%20types&ef_id=CjwKCAjwi8iXBhBeEiwAKbUofUpKM9nHToU9fsBJKApR3ccQzKs3LxSJ97PKiW5SvFRFwW6BnYP5xxoCOTEQAvD_BwE:G:s&s_kwcid=AL!4422!3!536392622533!e!!g!!aws%20instance%20types#Accelerated_Computing) EC2 instances

    ```{eval-rst}
    .. note::

        We use a `p4d.24xlarge` instance in this tutorial.  However these steps should run on any P-type EC2 instance.

    ```

- Download the latest [Imagenet Dataset](http://www.image-net.org/)

    ```{eval-rst}
    .. note::

        Due to the challenges associated with distributing ImageNet, we assume users to provide their own version of this dataset for the purpose of this tutorial.

    ```

- [MosaicML's ResNet-50 Recipes Docker Image](https://hub.docker.com/r/mosaicml/pytorch_vision/tags)
    - Tag: `mosaicml/pytorch_vision:resnet50_recipes`
    - The image comes pre-configured with the following dependencies:
        - Mosaic ResNet Training recipes
        - Training entrypoint: `train.py`
        - Composer Version: [0.9.0](https://github.com/mosaicml/composer/tree/v0.9.0)
        - PyTorch Version: 1.11.0
        - CUDA Version: 11.3
        - Python Version: 3.9
        - Ubuntu Version: 20.04

## Launching an AWS EC2 Instance

First let’s create an EC2 instance that we can run GPU training on.

1. Login to your AWS account and open the Management Console
1. For the purposes of this material, we will configure and launch a new `p4d.24xlarge` instance.  On your `EC2 Dashboard` click the `Launch instance` button.

    ![Launch Instance](https://storage.googleapis.com/docs.mosaicml.com/images/tutorials/launch_instance.png)

1. Name your instance and select an AMI, Instance type, Key pair and Network settings.  The following settings were used for this tutorial, customize as required depending on your AWS setup and IT requirements:
    - Name: `composer-r50-demo-a100x8`
    - Amazon Machine Image (AMI): `Deep Learning AMI GPU PyTorch 1.12.0 (Amazon Linux 2)`
    - Instance type: `p4d.24xlarge`
    - Key pair: `Create key pair` (make sure to note where you save the private key)
        - Key pair name: `composer_demo`
        - Key pair type: `RSA`
        - Private key format: `.pem`
    - Network settings: Use defaults
    - Storage (volumes):

        ![EBS Configuration](https://storage.googleapis.com/docs.mosaicml.com/images/tutorials/configure_ebs.png)

1. Click `Launch instance`!

## Configuring your AWS EC2 instance

Next we will connect to our newly launched `p4d.24xlarge` instance, perform some basic system configuration to optimize our runtime environment and setup our dataset area.

1. Navigate back to the `Instances` page in your AWS console.  Click on the running instance you just launched and in the Details pane, copy the instance’s `Public IPv4 DNS` address.  You will need this value to connect to the instance.
1. Using the private key you downloaded during the launch configuration and the instance’s public DNS address, connect to the system using SSH:

    <!--pytest.mark.skip-->
    ```bash
    ssh -i <path_to_private_key> ec2-user@<public_dns_address>
    ```

    For example,

    <!--pytest.mark.skip-->
    ```bash
    ssh -i ~/composer_demo.pem ec2-user@ec2-11-222-333-44.us-west-2.compute.amazon.com
    ```

1. Now let’s create a `datasets` area to place the ImageNet data as follows:

    <!--pytest.mark.skip-->
    ```bash
    sudo mkdir -p /datasets/ImageNet
    sudo chmod -R 777 /datasets
    ```

    (Optional )If the EC2 instance you selected comes direct attached [Instance Store Volumes](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/add-instance-store-volumes.html), it can be mounted as follows:

    <!--pytest.mark.skip-->
    ```bash
    sudo mkfs -t xfs /dev/nvme1n1
    sudo mkdir ImageNet
    sudo mount /dev/nvme1n1 /ImageNet
    sudo chmod 777 ImageNet/
    ```

    Instance Store Volumes (ISV) generally have better performance than EBS volumes since they are directly attached to the instance, at the expense of persistence.  Thus Instance Store Volumes are ephemeral and any data stored on these volumes will be inaccessible after the instance is powered off.

    Regardless of whether you choose to use an EBS volume or Instance Store Volume to host your dataset, the ImageNet data can be copied to the `/datasets/Imagenet` folder.  In our example, the directory tree under `/datasets` looks as follows:

    <!--pytest.mark.skip-->
    ```bash
    [ec2-user@ip-172-31-0-30 /]$ find ./datasets/ -maxdepth 2
    ./datasets/
    ./datasets/imagenet_files.tar
    ./datasets/ImageNet
    ./datasets/ImageNet/train
    ./datasets/ImageNet/val
    ```

    Once you populate the dataset area, you’ll be ready to start training!

## Train ResNet-50 on ImageNet

Now that we have launched an EC2 instance, configured the runtime and populated the dataset area, we are ready to kick off training.

1. Pull and run the `mosaicml/pytorch_vision:resnet50_recipes` Docker image.  The image contains everything required to train including: pre-installed Composer, package dependencies, training entrypoint and recipe configuration files.

    <!--pytest.mark.skip-->
    ```bash
    docker run -it -v /datasets:/datasets --gpus all --shm-size 1g mosaicml/pytorch_vision:resnet50_recipes
    ```

    ```{eval-rst}
    .. note::

        The default shared memory size of a Docker container is typically too small for larger datasets.  In this example, increasing the shared memory size to 1GB is usually sufficient.

    ```

2. Run ResNet-50 Training using the Mild recipe!

    <!--pytest.mark.skip-->
    ```bash
    composer train.py -f recipes/resnet50_mild.yaml --scale_schedule_ratio 0.36 \
        --train_dataset.imagenet.ffcv_dir /datasets/ImageNet/ffcv                     \
        --val_dataset.imagenet.ffcv_dir /datasets/ImageNet/ffcv
    ```

    ```{eval-rst}
    .. note::

        The ResNet-50 Mild and Medium recipes utilize the very efficient and high performing [FFCV dataloader](https://ffcv.io/), requiring the raw ImageNet data to be processed into FFCV format.  Composer can automatically perform this step for you prior to launching the training run, simply append the following command line arguments to the training command above:
        `--train_dataset.imagenet.datadir /datasets/ImageNet/`
        `--val_dataset.imagenet.datadir /datasets/ImageNet/`
        `--train_dataset.imagenet.ffcv_write_dataset`
        --val_dataset.imagenet.ffcv_write_dataset`
        The first two arguments simply specify the area of the raw ImageNet training and validation data, respectively.  The second two arguments enable dataset conversion if the expected FFCV formatted files do not exist.

        To perform this conversion manually, please follow the instructions detailed in the [README](https://github.com/mosaicml/examples/tree/main/examples/resnet#using-mosaic-recipes) in our [examples repository](https://github.com/mosaicml/examples/tree/main/examples/resnet), which contains all the code associated with our original blog post.

    ```


## Expected Results

We’ve performed various sweeps on AWS EC2 instances to understand the efficiency frontier across time, accuracy and cost as shown below.

![ResNet-50 on AWS, Explorer Runs](https://storage.googleapis.com/docs.mosaicml.com/images/tutorials/r50_aws_explorer.png)

The recipe explored in this tutorial should result in a model trained to a Top-1 accuracy of 76.6% in about 27 minutes for a total cost of $14.77.

![ResNet-50 on AWS, Explorer Recipe](https://storage.googleapis.com/docs.mosaicml.com/images/tutorials/r50_aws_explorer_recipe.png)

You can explore the results of our other [ResNet-50 runs on AWS](https://explorer.mosaicml.com/imagenet?sortBy=costSameQuality&model=resnet50&cloud=aws&hardware=all&algorithms=all&baseline=r50_optimized_p4d&recipe=mosaicml_baseline&recipe=mosaicml_hot&recipe=mosaicml_medium&recipe=mosaicml_mild) in [Explorer](https://explorer.mosaicml.com/), our tool for exploring efficiency frontiers for different models and datasets with different speed-up techniques across various clouds.

## Next steps

- [Explore other ResNet-50 recipes on AWS using Explorer!](https://explorer.mosaicml.com/imagenet?sortBy=costSameQuality&model=resnet50&cloud=aws&hardware=all&algorithms=all&baseline=r50_optimized_p4d&recipe=mosaicml_baseline&recipe=mosaicml_hot&recipe=mosaicml_medium&recipe=mosaicml_mild)
- Check out our [GitHub repository](https://github.com/mosaicml/composer) for the latest information on Composer
- Check out [Composer + FFCV: Faster Together](https://www.mosaicml.com/blog/composer-ffcv-faster-together) blog post for more information on how FFCV and Composer work together
- Reproduce our record setting [MLPerf ResNet-50 benchmark](https://www.mosaicml.com/blog/mlperf-2022)!  Note, you will require access to the `p4de.24xlarge` ([in preview](https://aws.amazon.com/ec2/instance-types/?trk=36c6da98-7b20-48fa-8225-4784bced9843&sc_channel=ps&sc_campaign=acquisition&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Compute|EC2|US|EN|Text&s_kwcid=AL!4422!3!536392622533!e!!g!!aws%20instance%20types&ef_id=CjwKCAjwi8iXBhBeEiwAKbUofUpKM9nHToU9fsBJKApR3ccQzKs3LxSJ97PKiW5SvFRFwW6BnYP5xxoCOTEQAvD_BwE:G:s&s_kwcid=AL!4422!3!536392622533!e!!g!!aws%20instance%20types)) EC2 instances which contain the Nvidia A100 80GB GPUs.  Please see the [MLPerf Training Results v2.0 GitHub Repository](https://github.com/mlcommons/training_results_v2.0/tree/main/MosaicML) for additional details.
- Try training on your models and datasets using Composer!
