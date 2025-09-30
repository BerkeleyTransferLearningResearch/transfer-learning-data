## Transfer Learning Dataset Overview

Transfer learning dataset is a domain adaptation dataset that contains data from three different domains:
- **Amazon**: Product images downloaded from the Amazon website  
- **DSLR**: Images taken with a digital single-lens reflex camera  
- **Webcam**: Images taken with a webcam  

## Dataset Statistics

### Overall Statistics
| Dataset | Number of Samples | Proportion |
|---------|-------------------|------------|
| Amazon  | 2,817             | 68.5%      |
| DSLR    | 498               | 12.1%      |
| Webcam  | 795               | 19.4%      |
| **Total** | **4,110**       | **100%**   |

### Category Distribution
The dataset contains 31 item categories, with the following distribution across the three domains:

| Category Name | Amazon | DSLR | Webcam | Total |
|---------------|--------|------|--------|-------|
| back_pack | 92 | 12 | 29 | 133 |
| bike | 82 | 21 | 21 | 124 |
| bike_helmet | 72 | 24 | 28 | 124 |
| bookcase | 82 | 12 | 12 | 106 |
| bottle | 36 | 16 | 16 | 68 |
| calculator | 94 | 12 | 31 | 137 |
| desk_chair | 91 | 13 | 40 | 144 |
| desk_lamp | 97 | 14 | 18 | 129 |
| desktop_computer | 97 | 15 | 21 | 133 |
| file_cabinet | 81 | 15 | 19 | 115 |
| headphones | 99 | 13 | 27 | 139 |
| keyboard | 100 | 10 | 27 | 137 |
| laptop_computer | 100 | 24 | 30 | 154 |
| letter_tray | 98 | 16 | 19 | 133 |
| mobile_phone | 100 | 31 | 30 | 161 |
| monitor | 99 | 22 | 43 | 164 |
| mouse | 100 | 12 | 30 | 142 |
| mug | 94 | 8 | 27 | 129 |
| paper_notebook | 96 | 10 | 28 | 134 |
| pen | 95 | 10 | 32 | 137 |
| phone | 93 | 13 | 16 | 122 |
| printer | 100 | 15 | 20 | 135 |
| projector | 98 | 23 | 30 | 151 |
| punchers | 98 | 18 | 27 | 143 |
| ring_binder | 90 | 10 | 40 | 140 |
| ruler | 75 | 7 | 11 | 93 |
| scissors | 100 | 18 | 25 | 143 |
| speaker | 99 | 26 | 30 | 155 |
| stapler | 99 | 21 | 24 | 144 |
| tape_dispenser | 96 | 22 | 23 | 141 |
| trash_can | 64 | 15 | 21 | 100 |

## Dataset Feature Analysis

### 1. Amazon Dataset
- **Number of Samples**: 2,817 images  
- **Characteristics**:
  - Largest number of samples, accounting for 68.5% of the total  
  - Relatively high image quality, clean backgrounds  
  - Relatively standardized product display angles  
- **Advantages**: Large dataset, suitable as a source domain for pre-training  
- **Disadvantages**: Large differences from real-world shooting environments  

### 2. DSLR Dataset
- **Number of Samples**: 498 images  
- **Characteristics**:
  - Smallest number of samples, only 12.1% of the total  
  - Captured with professional cameras, highest image quality  
- **Advantages**: High quality images, rich details  
- **Disadvantages**: Small dataset size, potential risk of overfitting  

### 3. Webcam Dataset
- **Number of Samples**: 795 images  
- **Characteristics**:
  - Medium-sized dataset, accounting for 19.4% of the total  
  - Captured with webcams, relatively lower image quality  
- **Advantages**: Closer to real-world application scenarios  
- **Disadvantages**: Relatively lower image quality, more noise  

## Category Imbalance Analysis

### Sample Count Distribution
- **Most Samples**: monitor (164 images)  
- **Fewest Samples**: bottle (68 images)  
- **Average Samples per Category**: 132.6 images  

### Inter-Domain Distribution Differences
1. **Amazon Domain**: Most categories have 90–100 samples, relatively balanced  
2. **DSLR Domain**: Sample counts range from 7–31, larger variation  
3. **Webcam Domain**: Sample counts range from 11–43, relatively balanced  
