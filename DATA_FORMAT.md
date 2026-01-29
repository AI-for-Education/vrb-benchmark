# Data Format Specification

This document describes the data format required for running evaluations with VRB Benchmark.

## Directory Structure

```
data/
├── options.csv                        # Answer format definitions (required)
├── <dataset>-questions-gold.csv       # Questions for a dataset (required)
└── <dataset>-images/                  # Images for a dataset (required)
    ├── Q001.png
    ├── Q002.png
    └── ...
```

Replace `<dataset>` with your dataset name (e.g., `test`, `zambia`, `india`).

## File Formats

### 1. options.csv

Defines the possible answer formats used across your questions.

**Location**: `data/options.csv`

**Format**:
```csv
option_id,values
alpha_upper,"A,B,C,D"
alpha_lower,"a,b,c,d"
numeric,"1,2,3,4"
```

**Columns**:
- `option_id` (string): Unique identifier for this option format
- `values` (string): Comma-separated list of valid answers (must be quoted)

**Common option formats**:
- `alpha_upper`: A, B, C, D (most common for multiple choice)
- `alpha_lower`: a, b, c, d
- `numeric`: 1, 2, 3, 4
- Custom: You can define any format (e.g., "Option1,Option2,Option3")

**Example**:
```csv
option_id,values
alpha_upper,"A,B,C,D"
alpha_lower,"a,b,c,d"
numeric,"1,2,3,4"
true_false,"True,False"
yes_no,"Yes,No"
```

### 2. Questions CSV

Contains your benchmark questions.

**Location**: `data/<dataset>-questions-gold.csv`

**Format**:
```csv
id,question,option_id,solution
Q001,How many circles are in this image?,alpha_upper,B
Q002,What color is the largest object?,alpha_upper,C
```

**Columns**:
- `id` (string, required): Unique identifier for the question. Must match the image filename (without extension).
- `question` (string, required): The question text that will be shown to the model.
- `option_id` (string, required): Reference to the answer format in `options.csv` (e.g., "alpha_upper").
- `solution` (string, optional): The correct answer (e.g., "A", "B"). Used for scoring but not required to run evaluations.

**Important**:
- The `id` must match exactly with the image filename (e.g., `Q001` → `Q001.png`)
- The `option_id` must exist in `options.csv`
- Question text should be clear and self-contained

**Example**:
```csv
id,question,option_id,solution
Q001,Count the number of red circles in the image.,alpha_upper,C
Q002,Which geometric shape appears most frequently?,alpha_upper,A
Q003,What is the relationship between objects A and B?,alpha_lower,b
```

### 3. Images

Visual content for each question.

**Location**: `data/<dataset>-images/`

**Naming convention**: `{question_id}.png`
- The filename (without extension) must match the `id` column in your questions CSV
- Example: Question `Q001` → Image `Q001.png`

**Format**:
- Supported formats: PNG (recommended), JPG, JPEG
- Images can be any size (will be sent to the model at high detail)
- Ensure images clearly show the answer options if they're embedded in the image

**Recommendations**:
- Use high-quality images (models perform better with clear images)
- If answer options are shown in the image, label them clearly (A, B, C, D)
- Keep image file sizes reasonable (models have context limits)
- Use consistent image dimensions within a dataset

## Question Design

### Prompt Template

The benchmark uses this template to format questions:

```
Please answer the following question about the image.
{question}

The image contains four possible answers labelled A, B, C, D.
Please think step by step and consider each option carefully.
Give your final answer option (e.g. one of A, B, C, D) as a single character on a new line at the end of your response.
```

This means:
- Your question text goes in the `{question}` placeholder
- The model will be instructed to select from the options defined by `option_id`
- The model will see both your question text AND the image

### Answer Options

You have two main approaches:

**1. Options in the image (recommended)**:
- Include the answer options visually in the image
- Label them clearly (A, B, C, D)
- The prompt template mentions options are "labelled" in the image
- Example: An image showing 4 different diagrams labeled A, B, C, D

**2. Options in the question text**:
- Include options in your question text
- Example: "What color is the largest object? A) Red, B) Blue, C) Green, D) Yellow"
- The model will still see the generic prompt about options being labeled

## Example Dataset

Here's a complete minimal example:

**data/options.csv**:
```csv
option_id,values
alpha_upper,"A,B,C,D"
```

**data/example-questions-gold.csv**:
```csv
id,question,option_id,solution
Q001,How many circles are visible in this image?,alpha_upper,B
Q002,What geometric pattern is shown?,alpha_upper,C
```

**data/example-images/**:
- `Q001.png` - Image showing 2 circles with options A=1, B=2, C=3, D=4 labeled
- `Q002.png` - Image showing a pattern with options A, B, C, D labeled

## Validation Checklist

Before running your benchmark, verify:

- [ ] `data/options.csv` exists and has at least one option format defined
- [ ] `data/<dataset>-questions-gold.csv` exists
- [ ] All `option_id` values in questions CSV exist in `options.csv`
- [ ] All questions have unique `id` values
- [ ] For each question `id`, a corresponding image exists: `data/<dataset>-images/{id}.png`
- [ ] Image filenames match question IDs exactly (case-sensitive)
- [ ] All images can be opened and are valid image files

## Testing Your Data

Use this script to verify your data loads correctly:

```python
from src.vrb_benchmark.load import load_questions, get_image_path, load_options
import os

# Load questions
df = load_questions(dataset="your-dataset-name")
print(f"Loaded {len(df)} questions")

# Check images exist
options = load_options("data/options.csv")
for _, row in df.iterrows():
    img_path = get_image_path(row["id"], dataset="your-dataset-name")
    if not os.path.exists(img_path):
        print(f"WARNING: Missing image for {row['id']}: {img_path}")
    if row["option_id"] not in options:
        print(f"WARNING: Unknown option_id '{row['option_id']}' for {row['id']}")

print("Validation complete!")
```

## Privacy Considerations

If your benchmark data is private:

1. Add `data/` to `.gitignore` (already done by default)
2. Consider providing:
   - A sample/dummy dataset for testing
   - Clear documentation on the expected data format
   - Instructions for users to prepare their own data

This allows others to use your benchmark framework with their own data.
