from google.api_core import operations_v1
from google.cloud.aiplatform_v1beta1 import FeaturestoreOnlineServingServiceClient, FeaturestoreServiceClient, FeatureSelector
from google.cloud.aiplatform_v1beta1.types import featurestore_online_service as featurestore_online_service_pb2
from google.cloud.aiplatform_v1beta1.types import entity_type as entity_type_pb2
from google.cloud.aiplatform_v1beta1.types import feature as feature_pb2
from google.cloud.aiplatform_v1beta1.types import featurestore as featurestore_pb2
from google.cloud.aiplatform_v1beta1.types import featurestore_service as featurestore_service_pb2
from google.cloud.aiplatform_v1beta1.types import io as io_pb2
from google.cloud.aiplatform_v1beta1.types import ListFeaturestoresRequest, CreateFeaturestoreRequest, Featurestore

from google.protobuf.timestamp_pb2 import Timestamp
from google.cloud.aiplatform_v1beta1.types import featurestore_monitoring as featurestore_monitoring_pb2
from google.protobuf.duration_pb2 import Duration



def create_fs(project, region, store_id, store_name=None):
    
    API_ENDPOINT = f"{region}-aiplatform.googleapis.com"  
    admin_client = FeaturestoreServiceClient(client_options={"api_endpoint": API_ENDPOINT})
    
    base_path = admin_client.common_location_path(project, region)
    
    for f in admin_client.list_featurestores(ListFeaturestoresRequest(parent=admin_client.common_location_path(project, region))):
        existing_id = f.name.split('/')[-1]
        if store_id == existing_id:
            print(f'Feature Store "{store_id}" already exists in {region}')
            return
    
    if store_name is None:
        store_name = f'{base_path}/{store_id}'
    
    req = CreateFeaturestoreRequest(
        parent = base_path,
        featurestore = Featurestore(
            name=store_name,
            online_serving_config=Featurestore.OnlineServingConfig(fixed_node_count=3)),
        featurestore_id = store_id)
    
    lro = admin_client.create_featurestore(req)
    name = lro.result()
    print(f'Created Feature Store {name} in {region}')
    return name


def create_entity(project, region, store_id, entity, entity_descr, features, features_descr=None):
    
    API_ENDPOINT = f"{region}-aiplatform.googleapis.com"  
    admin_client = FeaturestoreServiceClient(client_options={"api_endpoint": API_ENDPOINT})

    if features_descr is None:
        features_descr = features
    
    if len(features) != len(features_descr):
        print(f'ERROR: Got {len(features)} features and {len(features_descr)} descriptions')
        return
    
    print(f'Creating entity {entity} in Feature Store {store_id} ({region})')
    
    snapshot_analysis = featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                    monitoring_interval=Duration(seconds=3600))  # 1 hour
    
    lro = admin_client.create_entity_type(
        featurestore_service_pb2.CreateEntityTypeRequest(
            parent=admin_client.featurestore_path(project, region, store_id),
            entity_type_id=entity,
            entity_type=entity_type_pb2.EntityType(
             description=entity_descr,
             monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                snapshot_analysis=snapshot_analysis))
        )
    ).result()
    
    print(lro)
    
    def _create_f_request(name, descr):
        return featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description=descr,
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=snapshot_analysis)),
                feature_id=name)
    
    requests = [_create_f_request(x[0], x[1]) for x in zip(features, features_descr)]
    
    print(f'\nCreating features: {",".join(features)}')

    lro = admin_client.batch_create_features(
        parent=admin_client.entity_type_path(PROJECT, REGION, FEATURESTORE_ID, entity),
        requests=requests).result()
    
    return lro


def ingest_entities_csv(project, region, store_id, entity, features, gcs_uris):
    
    API_ENDPOINT = f"{region}-aiplatform.googleapis.com"  
    admin_client = FeaturestoreServiceClient(client_options={"api_endpoint": API_ENDPOINT})

    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    timestamp.nanos = 0
    
    specs = [featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(id=f) for f in features]
    
    import_request_transaction = featurestore_service_pb2.ImportFeatureValuesRequest(
        entity_type=admin_client.entity_type_path(project, region, store_id, entity),
        csv_source=io_pb2.CsvSource(gcs_source=io_pb2.GcsSource(uris=gcs_uris)),
        feature_specs=specs,
        entity_id_field=entity,
        feature_time=timestamp, # unique timestamp for all
        worker_count=5)
    
    print(f'Ingesting features for "{entity}" entity...')
    ingestion_lro = admin_client.import_feature_values(import_request_transaction).result()
    print('done')
    
    return ingestion_lro


# entity is the name of the entity type you want to read, for example: user
# entity_value is the specific instance of the entity that you want to have the feature of, for example a user ID
def read_features(project, region, store_id, entity, features, entity_value):
    
    API_ENDPOINT = f"{region}-aiplatform.googleapis.com"  
    admin_client = FeaturestoreServiceClient(client_options={"api_endpoint": API_ENDPOINT})
    data_client = FeaturestoreOnlineServingServiceClient(client_options={"api_endpoint": API_ENDPOINT})    
    
    feature_selector = FeatureSelector()
    feature_selector.id_matcher.ids = features
    
    read_request = featurestore_online_service_pb2.ReadFeatureValuesRequest(
        entity_type = admin_client.entity_type_path(project, region, store_id, entity),
        entity_id = entity_value,
        feature_selector=feature_selector)
    
    res = data_client.read_feature_values(read_request)
    values = [d.value for d in res.entity_view.data]

    # return a dict with { 'feature1': val1, 'feature2': val2, ... }
    # exclude features that do not have a generate_time: these do not exist in the store
    return {f:v.double_value for (f,v) in zip(features, values) if v.metadata.generate_time}