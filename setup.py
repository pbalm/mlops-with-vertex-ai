import setuptools


REQUIRED_PACKAGES_AS_IMG = [
    "kfp==1.8.12",
    "google-cloud-bigquery==2.34.3",
    "google-cloud-bigquery-storage==2.13.2",
    "google-cloud-aiplatform==1.14.0",
    "google-cloud-pubsub",
    "cloudml-hypertune==0.1.0.dev6",
    "pytest==7.1.2",
    #tensorflow==2.8.2
    "tensorflow-data-validation==1.8.0",
    "tensorflow-transform==1.8.0",
    "tfx==1.8.0",
    "tensorflow-io==0.26.0",
    "apache-beam[gcp]==2.39.0"
]

REQUIRED_PACKAGES_UPDATED = [
    "google-cloud-aiplatform==1.14.0",
    "tensorflow-transform==1.8.0",
    "tensorflow-data-validation==1.8.0",
    "cloudml-hypertune==0.1.0.dev6",
    "tfx-bsl==1.8.0"
]

REQUIRED_PACKAGES_ORI = [
    "google-cloud-aiplatform==1.4.2",
    "tensorflow-transform==1.2.0",
    "tensorflow-data-validation==1.2.0",
    "cloudml-hypertune==0.1.0.dev6"
]


setuptools.setup(
    name="executor",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES_ORI,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"src": ["raw_schema/schema.pbtxt"]},
)
