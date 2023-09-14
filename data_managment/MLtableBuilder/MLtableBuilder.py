
# TODO: check if the image urls are valid
# env variables: AZURE_CLIENT_ID, AZURE_CLIENT_ID, AZURE_TENANT_ID, RESOURCE_GROUP, SUBSCRIPTION_ID

import os
import json
from datetime import datetime
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential, DefaultAzureCredential

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

class MLtableBuilder:
    def __init__(self, table_name:str, table_version:int=1,
                table_path:str='./mltable',
                 image_urls:list=None, Labels:list=None, tags:list=None, image_urls_column:str='image_url',
                 aml_workspace:str=None, resource_group:str=None, subscription_id:str=None, 
                streaming = False):
        self.image_urls = image_urls
        self.Labels = Labels
        self.tags = tags
        self.table_name = table_name
        self.table_version = table_version
        self.aml_workspace = aml_workspace
        self.image_urls_column = image_urls_column
        if resource_group is None:
            self.resource_group = os.getenv("RESOURCE_GROUP")
        else:
            self.resource_group = resource_group
        if subscription_id is None:
            self.subscription_id = os.getenv("SUBSCRIPTION_ID")
        else:
            self.subscription_id = subscription_id
        self.table_path = table_path
        self.datastores_dict = {}
        self.streaming = streaming

        self.aml_client = self.connect_to_workspace()

        # url example: 'https://<storage_account_name>.blob.core.windows.net/<container_name>/<path>'
        # aml url example: 'azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/datastores/<datastore_name>/paths/<path>'
        # image_urls example: ['url', 'url']
        # Labels example: [{'task1': url, 'task2': 'url'}, {'task1': url, 'task3': 'url', 'task2': 'url'}]
        # tags example: [{'tag1': 'value1:string', 'tag2': 'value2:string'}, {'tag1': 'value1:string', 'tag2': 'value2:string'}]

        # list all the tasks in Labels
        self.tasks = []
        for label in self.Labels:
            for task in label:
                if task not in self.tasks:
                    self.tasks.append(task)

        # list all the tags in tags
        self.TAGS = []
        for tag in self.tags:
            for t in tag:
                if t not in self.TAGS:
                    self.TAGS.append(t)
        
        # check if the number of labels is the same as the number of images
        if len(self.image_urls) != len(self.Labels):
            raise ValueError("The number of labels is not the same as the number of images")
        # check if for eche Label there is at least one label
        for label in self.Labels:
            if len(label) == 0:
                raise ValueError("For each image there must be at least one label")
        
        # create a table name
        if table_name is None:
            self.table_name = "MLtable_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # connect to the workspace using service principal authentication and check if the workspace is connected
    def connect_to_workspace(self):
        try:
            credential = DefaultAzureCredential()
            aml_client = MLClient(credential=credential,
                                workspace_name=self.aml_workspace,
                                resource_group_name=self.resource_group,
                                subscription_id=self.subscription_id)
            
        except:
            # instantiate the MLClient using service principal authentication
            credential = ClientSecretCredential(client_id=os.environ["AZURE_CLIENT_ID"],
                                                client_secret=os.environ["AZURE_CLIENT_SECRET"],
                                                tenant_id=os.environ["AZURE_TENANT_ID"])

            aml_client = MLClient(credential=credential,
                                workspace_name=self.aml_workspace,
                                resource_group_name=self.resource_group,
                                subscription_id=self.subscription_id)
    
        # check if the workspace is connected
        if aml_client is None:
            raise ValueError("The workspace is not connected")
        else:
            return aml_client

    # check aml workspace has a table with the same name
    def check_table_name(self):
        try:
            tb = self.aml_client.data.get(self.table_name, version=str(self.table_version))
            raise ValueError(f"The table {self.table_name}:v{self.table_version} already exists")
        except:
            return True
        
        
    
    # TODO: check if the image urls are valid
    # check if aml workspace has access to the image urls
    def check_image_urls(self):
        # check if the image urls are valid
        for url in self.image_urls:
            if not os.path.exists(url):
                raise ValueError("The image url is not valid")
        return True
    
    # TODO: create a new datastore
    def create_datastore(self, storage_account_name:str, container_name:str):
        """Create a new datastore"""
        # get the storage account key
        # storage_account_key = 
        # create a new datastore
        # self.aml_client.datastores.create_or_update
        # (storage_account_name=storage_account_name,
        #                                 storage_account_key=storage_account_key,
        #                                 container_name=container_name)
        pass

    # Convert image urls to aml urls
    def convert_image_urls_to_aml_urls(self, img_url):
        
        # get the storage account name and the container name
        # url example: 'https://<storage_account_name>.blob.core.windows.net/<container_name>/<path>'
        storage_account_name = img_url.split(".")[0].split("//")[1]
        container_name = img_url.split("//")[-1].split("/")[1]
        img_path = '/'.join(img_url.split("//")[-1].split("/")[2:])
        # check if f"{storage_account_name}_{container_name}" is a key in datastores_dict
        if f"{storage_account_name}_{container_name}" not in self.datastores_dict:
            # find the datastore with the same storage account name and container name
            for datastore in self.aml_client.datastores.list():
                if datastore.type == "AzureBlob":
                    if datastore.account_name == storage_account_name and datastore.container_name == container_name:
                        self.datastores_dict[f"{storage_account_name}_{container_name}"] = datastore.name
                        break
        # check if f"{storage_account_name}_{container_name}" is a key in datastores_dict
        if f"{storage_account_name}_{container_name}" not in self.datastores_dict:
            # TODO: create a new datastore
            # self.create_datastore(storage_account_name=storage_account_name, container_name=container_name)
            raise ValueError("The datastore does not exist")

        # add the image url
        # image_url example: azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/datastores/<datastore_name>/paths/<path>
        datastore_name = self.datastores_dict[f"{storage_account_name}_{container_name}"]
        image_url = f"azureml://subscriptions/{self.subscription_id}/resourcegroups/{self.resource_group}/workspaces/{self.aml_workspace}/datastores/{datastore_name}/paths/{img_path}"
        return image_url
                
    # create a table
    def create_jsonl(self):
        """Create a jsonl file"""

        # check if self.table_path exists otherwise create it
        os.makedirs(self.table_path, exist_ok=True)
        self.table_jsonl = os.path.join(self.table_path, self.table_name + ".jsonl")
        # create a jsonl file
        
        with open(self.table_jsonl, "w") as f:
            for i in range(len(self.image_urls)):
                # create a dictionary for each image
                d = {}
                d[self.image_urls_column] = self.convert_image_urls_to_aml_urls(self.image_urls[i])
                # add the labels
                for label in self.Labels[i]:
                    if self.Labels[i][label] != None:
                        d[label] = self.convert_image_urls_to_aml_urls(self.Labels[i][label])
                    else:
                        d[tag] = ""
                # add the tags
                for tag in self.tags[i]:
                    if self.tags[i][tag] != None:
                        d[tag] = self.tags[i][tag]
                    else:
                        d[tag] = ""
                # write the dictionary to the jsonl file
                f.write(json.dumps(d) + "\n")
        print(f"The jsonl file is created: {self.table_jsonl}")
        return True
    
    
    def create_ml_table_file(self):
        """Create ML Table definition"""
        
        if self.streaming:

            MLTable="paths:\n" + \
                   f"  - file: ./{self.table_name}.jsonl\n" + \
                    "transformations:\n"    + \
                    "  - read_json_lines:\n" + \
                    "        encoding: utf8\n" + \
                    "        invalid_lines: error\n" + \
                    "        include_path_column: false\n" + \
                    "  - convert_column_types:\n"   + \
                   f"      - columns: {self.image_urls_column} \n"  + \
                    "        column_type: stream_info \n" 

            # assume that the labels are stream_info
            for task in self.tasks:
                MLTable = MLTable + "      - columns: " + task + "\n" + \
                                    "        column_type: stream_info  \n"

            # assume that the tags are string
            for tag in self.TAGS:
                MLTable = MLTable + "      - columns: " + tag + "\n" + \
                                    "        column_type: string  \n"   
         
        else:
            MLTable="paths:\n" + \
                   f"  - file: ./{self.table_name}.jsonl\n" + \
                    "transformations:\n"    + \
                    "  - read_json_lines:\n" + \
                    "        encoding: utf8\n" + \
                    "        invalid_lines: error\n" + \
                    "        include_path_column: false\n" + \
                    "  - convert_column_types:\n"   + \
                   f"      - columns: {self.image_urls_column} \n"  + \
                    "        column_type: string \n" 

            # assume that the labels are stream_info
            for task in self.tasks:
                MLTable = MLTable + "      - columns: " + task + "\n" + \
                                    "        column_type: string  \n"

            # assume that the tags are string
            for tag in self.TAGS:
                MLTable = MLTable + "      - columns: " + tag + "\n" + \
                                    "        column_type: string  \n" 
            
        
        with open(os.path.join(self.table_path, "MLTable"), "w") as f:
            f.write(MLTable)


    # create a azure machine learning mltable dataset
    def create_aml_table(self):
        """
        mltable = Data(
            path=my_path,
            type=AssetTypes.MLTABLE,
            description="<description>",
            name="<name>",
            version='<version>'
        )

        ml_client.data.create_or_update(my_data)
        """
        # create a MLTable dataset
        mltable = Data(
            path=self.table_path,
            type=AssetTypes.MLTABLE,
            description="MLTable dataset",
            name=self.table_name,
            version=str(self.table_version)
        )
        return mltable


    # create jsonl and MLTable files and upload them to the workspace
    def upload_table(self):
        # check if the table name is already in use
        self.check_table_name()
        # check if the image urls are valid
        # self.check_image_urls() TODO: check if the image urls are valid
        # create jsonl and MLTable files
        self.create_jsonl()
        self.create_ml_table_file()
        # upload the jsonl and MLTable files to the workspace
        mltable = self.create_aml_table()
        self.aml_client.data.create_or_update(mltable)

    
        
    