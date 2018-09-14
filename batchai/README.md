# Batch AI setup instructions

## Create resources

```bash
BATCHAI_SA=<storage-account-name>
BATCHAI_RG=<resource-group-name>
BATCHAI_WS=rnnbaiws <workspace-name>
BATCHAI_CLUST=<cluster-name>
BATCHAI_EXP=<experiment-name>

az group create -l southcentralus -n ${BATCHAI_RG}
az batchai workspace create -l southcentralus -g ${BATCHAI_RG} -n ${BATCHAI_WS}

az storage account create -n ${BATCHAI_SA} --sku Standard_LRS -g ${BATCHAI_RG}

az storage share create -n logs --account-name ${BATCHAIR_RG}
az storage share create -n resources --account-name ${BATCHAIR_SA}
az storage share create -n output --account-name ${BATCHAIR_SA}
az storage directory create -n scripts -s resources --account-name ${BATCHAIR_SA}
az storage directory create -n data -s resources --account-name ${BATCHAIR_SA}
az storage file upload -s resources --source energy.csv --path data --account-name ${BATCHAIR_SA}

az batchai cluster create -g ${BATCHAI_RG} -w ${BATCHAI_WS} -n ${BATCHAI_CLUST} --user <username> --password <password> --image UbuntuLTS --vm-size Standard_NC6 --max 10 --min 1 --storage-account-name ${BATCHAI_SA}
```

## Test docker image
```bash
cd batchai/docker
# start docker container in interactive mode
CID="$(nvidia-docker run -v $(pwd):/rnntutorial -dit angusrtaylor/rnntutorial)"
# start bash session in container
nvidia-docker exec -it ${CID} bash
```
## Run Batch AI jobs

```bash
# create experiment
az batchai experiment create -g ${BATCHAI_RG} -w ${BATCHAI_WS} -n ${BATCHAI_EXP}
# create job
az batchai job create -c ${BATCHAI_CLUST} -n job1 -g ${BATCHAI_RG} -w ${BATCHAI_WS} -e ${BATCHAI_EXP} -f job.json --storage-account-name ${BATCHAI_SA}
```