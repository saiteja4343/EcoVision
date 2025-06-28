# EcoVision

Heading
```mermaid
graph TD;
    A[Roboflow Dataset Bounding Boxes] --> B[SAM 2.1 Bbox to Segmentation];
    C[Classes with less images] --> D[Flux 1.dev Synthetic Generation];
    D --> E[Lang-SAM Synthetic Segmentation];
    B --> F[Combine Datasets];
    E --> F;
    F --> G[Data Preprocessing Balance & Split];
    G --> H[Final Dataset Train/Test/Valid];

```

