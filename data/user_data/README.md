# PRBEN Released Data

This directory contains the released benchmark data files for the PRBEN dataset.  
All original user identifiers have been anonymized into unified IDs in the following format:

```text
prben_id_xxxxx
```
---

# File Structure

## `id_to_query.jsonl`

Maps each anonymized ID to its corresponding search query.

### Example

```json
{
  "id": "prben_id_00001",
  "value": "query"
}
```

### Fields

| Field | Description |
|---|---|
| `id` | Anonymized sample ID |
| `value` | Original user query |

---

## `user_history_part*.jsonl`

Contains the user's historical search behavior sequence.

### Example

```json
{
  "id": "prben_id_00001",
  "value": [
    "best pillow for neck pain",
    "sleep meditation music",
    "melatonin side effects"
  ]
}
```

### Fields

| Field | Description |
|---|---|
| `id` | Anonymized sample ID |
| `value` | Historical search queries |

---

## `user_profile.jsonl`

Contains anonymized user profile information.

### Example

```json
{
  "id": "prben_id_00001",
  "value": {
    "gender": "男",
    "age": "0-16",
    "province": "北京"
  }
}
```

### Fields

| Field | Description |
|---|---|
| `id` | Anonymized sample ID |
| `gender` | User gender |
| `age` | User age group |
| `province` | User province |

### Notes

- All profile information has been anonymized and desensitized.
- No personally identifiable information (PII) is included.

---

## `user_click_url.jsonl`

Contains clicked URLs associated with each query.

### Example

```json
{
  "id": "prben_id_00001",
  "value": [
    "https://example.com/page1",
    "https://example.com/page2"
  ]
}
```

### Fields

| Field | Description |
|---|---|
| `id` | Anonymized sample ID |
| `value` | List of clicked URLs |

---



# Data Format

All files use the JSONL format.

- One JSON object per line
- UTF-8 encoding

### Loading Example

```python
import json

with open("id_to_query.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        print(item)
```



# Privacy & Ethics

- All user identifiers have been anonymized.
- Personally identifiable information has been removed.
- The dataset is released strictly for research purposes.
