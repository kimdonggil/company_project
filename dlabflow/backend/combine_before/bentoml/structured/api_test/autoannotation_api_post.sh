curl -X POST "http://10.40.217.236:4123/autoannotation/post" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
           "id": 111,
           "projectId": "cf6a5bcb-4259-41ed-9cff-6410beeedc26",
           "versionId": "ab2686ee-40f0-4f03-a80a-5b03aecac812",
           "datasetId": "2f5d7e6c-38b8-4a2d-9357-13b5e1e6821a",
           "autoAlgorithm": "yolo_version_8_small",
           "minConfidence": 0.1,
           "maxConfidence": 1.0,
           "targetImagePaths": [
              "grit/autoannotation-test/dataset/2007_000187_jpg.rf.2ab27aab673e7d40a76a53440f177c98.jpg",
              "grit/autoannotation-test/dataset/2007_000121_jpg.rf.807519c5d839ae8e10504ffb0c132f39.jpg",
              "grit/autoannotation-test/dataset/2007_000061_jpg.rf.46ef80849da87d8113c0330be6ea5beb.jpg",
              "grit/autoannotation-test/dataset/2007_000033_jpg.rf.83ab0d65cbcc0be92b649082c8a21ffb.jpg",
              "grit/autoannotation-test/dataset/2007_000032_jpg.rf.453cf71521fb73718369a7f07a41433c.jpg"
           ],
           "classDefinitions": [
              {"className": "bicycle", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "cat", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "bus", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "aeroplane", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "diningtable", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "sofa", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "boat", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "tvmonitor", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "sheep", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "person", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "motorbike", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "bottle", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "chair", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "train", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "dog", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "bird", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "horse", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "car", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "cow", "imageDatasetAnnotationTechnique": "rectanglelabels"},
              {"className": "pottedplant", "imageDatasetAnnotationTechnique": "rectanglelabels"}
           ]
        }'
