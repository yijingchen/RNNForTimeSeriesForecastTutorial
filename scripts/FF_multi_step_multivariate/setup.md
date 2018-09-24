# Environment Setup

## Log into Azure account

TBD

## Download and Extract dataset

TBD

## Create Batch AI resources

Run the following commands in bash shell:

```bash
BATCHAI_SA=<storage-account-name>
BATCHAI_RG=<resource-group-name>
BATCHAI_WS=<batch-ai-workspace-name>
BATCHAI_CLUST=<batch-ai-cluster-name>
BATCHAI_EXP=<batch-ai-experiment-name>

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

## Create service principal

TBD

## Create configuration file

TBD


