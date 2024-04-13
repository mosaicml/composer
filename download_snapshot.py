from composer.utils.object_store import OCIObjectStore

oci_client = OCIObjectStore(bucket="ning-test", prefix="mem_snapshot")

#objs = oci_client.list_objects()
#print(f"bigning debug objs = {objs}")


for rank in range(2):

    oci_client.download_object(
        object_name = f"mem_snapshot/snapshot_{rank}",
        filename = f"snapshot_{rank}.pickle",
        overwrite=True,
    )

    break
    oci_client.download_object(
        object_name = f"trace_{rank}",
        filename = f"trace_{rank}.html",
        overwrite=True,
    )
