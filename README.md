# Evaluation framework for 3D generative AI

Evaluations in 3D generative AI are broken. Let's fix them. In these evaluations, we assess images from realistic settings, including single objects, scenes and kits. But the key point is to let you bring your own images! Just drop them into the `images` folder and run `./run_eval.sh`. This will run various model permutations, including automatically producing a parts-based kit.

## Overview

This repository contains CSM.AI API documentation and evaluation scripts for testing various 3D generation capabilities. The evaluation framework includes **automatic retry handling** for failed jobs due to rate limits or temporary API issues. 

The `full_eval_set` folder has an extensive evaluation set and we will post qualitiative and quantitative results on this here (with continual updates).

Note: This script does not run AI retopology as this can be time consuming - retopology can be run separately on these outputs using the API docs at [https://docs.csm.ai/sessions/retopology](https://docs.csm.ai/sessions/retopology) (or simply from the UI; all assets ran from here will show up on [3d.csm.ai](https://3d.csm.ai) in your account for easy access and downloading).

## Quick Start

### Get Your API Key

Before using the evaluation scripts, you'll need a CSM.ai API key:

1. Go to [https://3d.csm.ai/](https://3d.csm.ai/)
2. Click on **Profile Settings** (bottom left)
3. Navigate to **Settings → Developer Settings**
4. Copy your API key

The evaluation script will prompt you for your API key on first run and **automatically save it locally** for future use.

### Running the Evaluation

```bash
# Show available commands
./run_eval.sh

# Run the full evaluation (recommended for first time)
./run_eval.sh run
```

### Available Commands

The `run_eval.sh` script provides several commands for managing the evaluation workflow:

| Command | Description | Usage |
|---------|-------------|-------|
| **`run`** | Full evaluation workflow (setup + evaluate all images in images/*) | `./run_eval.sh run` |
| **`setup`** | Setup environment and dependencies only | `./run_eval.sh setup` |
| **`eval`** | Run evaluation only (skip setup) | `./run_eval.sh eval` |
| **`progress`** | Check progress of submitted jobs | `./run_eval.sh progress` |
| **`clean`** | Clean up previous results (interactive) | `./run_eval.sh clean` |
| **`help`** | Show all commands and examples | `./run_eval.sh help` |

### Command Details

**🚀 `run`** - Complete workflow
- Sets up Python environment  
- Installs dependencies
- Creates directories
- Runs evaluation for all images in `images/`
- Shows summary

**⚙️ `setup`** - Environment preparation
- Checks Python/pip installation
- Installs required packages
- Creates `images/` and `results/` directories
- Warns if no images found

**🎯 `eval`** - Evaluation only
- Runs evaluation (assumes setup complete)
- Best for re-running after adding new images
- **Automatically retries failed jobs** that are eligible for retry
- Uses intelligent retry logic for temporary failures (rate limits, timeouts, server errors)
- Implements exponential backoff (2s, 4s, 8s, max 30s) to be respectful to the API
- Maximum of 3 retry attempts per job

**📊 `progress`** - Job monitoring  
- Checks status of previously submitted jobs
- Updates job tracking file
- Shows completion progress
- Run periodically to monitor 30-60min job processing
- **Note**: Job tracking uses image file names as unique identifiers

**🧹 `clean`** - Reset evaluation
- Interactive confirmation required
- Deletes `results/` directory and `job_tracking.json`
- Use for fresh start or troubleshooting

### Common Workflows

```bash
# See available commands
./run_eval.sh

# First time setup
./run_eval.sh run

# Check job progress (run every 15-30 minutes)  
./run_eval.sh progress

# Add new images and evaluate them
./run_eval.sh eval

# Fresh start (removes all previous results)
./run_eval.sh clean
./run_eval.sh run

# Troubleshooting
./run_eval.sh progress   # Check current status
./run_eval.sh eval       # Re-run evaluation (will retry any failed jobs automatically)
```

### Evaluation Jobs

For each image in `images/`, runs 6 job configurations with `resolution=200000`:

1. **Image-to-3D (base)**: `geometry_model='base'`
2. **Image-to-3D (turbo)**: `geometry_model='turbo'` 
3. **Image-to-3D (turbo + baked)**: `geometry_model='turbo' + texture_model='baked'`
4. **Image-to-3D (turbo + pbr)**: `geometry_model='turbo' + texture_model='pbr'`
5. **Image-to-Kit**: `decomposition_model='pro' + geometry_model='turbo' + texture_model='baked'`
6. **Chat-to-3D**: Re-prompt image, then Image-to-3D

See [`run_eval.sh help`](run_eval.sh) for all commands.

## File Structure

```
├── csm_eval.py              # Main evaluation script
├── job_tracking.json        # Job tracking (created automatically)
├── requirements.txt         # Python dependencies
├── USAGE.md                # Detailed usage guide
├── README.md               # This file
├── .csm_config             # API key storage (created automatically)
├── images/                 # Place your test images here
│   └── test1.png
└── results/                # Output directory (created automatically)
    ├── job_summary.json
    ├── combined_submission_results.json
    └── {image_name}_submission_results.json
```

---

<details>
<summary><strong>📚 API Reference Documentation</strong> (<a href="https://docs.csm.ai">Full Reference</a>)</summary>

### Image-to-3D

Convert a single image into a 3D model.

**Endpoint**: `POST https://api.csm.ai/v3/sessions/`

**JavaScript Example**:
```javascript
const url = 'https://api.csm.ai/v3/sessions/';
const options = {
	method: 'POST',
	headers: {'x-api-key': '123', 'Content-Type': 'application/json'},
	body: '{"type":"image_to_3d","input":{"image":"https://picsum.photos/200/300.jpg"}}'
};

try {
	const response = await fetch(url, options);
	const data = await response.json();
	console.log(data);
} catch (error) {
	console.error(error);
}
```

**Response Example**:
```json
{
  "_id": "SESSION_XXXXXXXXX_XXXXXXXX",
  "user_id": "XXXXXXXXXXXXXXXXXX",
  "status": "incomplete",
  "type": "image_to_3d",
  "input": {
    "image": {
      "_id": "ASSET_XXXXXXXXXX_XXXXXXX",
      "name": "",
      "status": "complete",
      "type": "image",
      "data": {
        "image_url": "https://example.com/image.jpg"
      }
    },
    "model": "sculpt",
    "settings": {
      "geometry_model": "base",
      "texture_model": "none",
      "topology": "tris",
      "resolution": 100000,
      "symmetry": "off",
      "scaled_bbox": [-1, -1, -1],
      "preserve_aspect_ratio": false,
      "pivot_point": [0, -0.5, 0]
    }
  },
  "output": {
    "segmented_image_url": "",
    "meshes": [
      {
        "_id": "ASSET_XXXXXXXXXX_XXXXXXX",
        "name": "",
        "status": "incomplete",
        "type": "mesh",
        "data": {
          "image_url": "",
          "glb_url": "",
          "obj_url": "",
          "fbx_url": "",
          "usdz_url": ""
        }
      }
    ]
  }
}
```

### Check Session Status

Monitor job progress and retrieve results when complete.

**Endpoint**: `GET https://api.csm.ai/v3/sessions/{session_id}`

**Parameters**:
- `session_id` (string, required): The session ID returned from session creation (starts with `SESSION_`)

**JavaScript Example**:
```javascript
const url = 'https://api.csm.ai/v3/sessions/{session_id}';
const options = {
	method: 'GET',
	headers: {'x-api-key': '123', 'Content-Type': 'application/json'}
};

try {
	const response = await fetch(url, options);
	const data = await response.json();
	console.log(data);
} catch (error) {
	console.error(error);
}
```

**Complete Response Example**:
```json
{
  "_id": "SESSION_XXXXXXXXX_XXXXXXXX",
  "user_id": "XXXXXXXXXXXXXXXXXX",
  "status": "complete",
  "type": "image_to_3d",
  "input": {
    "image": {
      "_id": "ASSET_XXXXXXXXXX_XXXXXXX",
      "name": "",
      "status": "complete",
      "type": "image",
      "data": {
        "image_url": "https://example.com/image.jpg"
      }
    },
    "num_variations": 1,
    "manual_segmentation": false,
    "model": "sculpt",
    "settings": {
      "geometry_model": "base",
      "texture_model": "none",
      "topology": "tris",
      "resolution": 100000,
      "symmetry": "off",
      "scaled_bbox": [-1, -1, -1],
      "preserve_aspect_ratio": false,
      "pivot_point": [0, -0.5, 0]
    }
  },
  "output": {
    "segmented_image_url": "https://example.com/segmented_image.png",
    "meshes": [
      {
        "_id": "ASSET_XXXXXXXXXX_XXXXXXX",
        "name": "",
        "status": "complete",
        "type": "mesh",
        "data": {
          "image_url": "https://example.com/image.jpg",
          "glb_url": "https://example.com/glb.glb",
          "obj_url": "https://example.com/obj.obj",
          "fbx_url": "https://example.com/fbx.fbx",
          "usdz_url": "https://example.com/usdz.usdz"
        }
      }
    ]
  }
}
```

### Image-to-Kit

Convert an image into multiple 3D parts/components.

**Endpoint**: `POST https://api.csm.ai/v3/sessions/`

**JavaScript Example**:
```javascript
const url = 'https://api.csm.ai/v3/sessions/';
const options = {
	method: 'POST',
	headers: {'x-api-key': '123', 'Content-Type': 'application/json'},
	body: '{"type":"image_to_kit","input":{"image":"https://example.com/image.png","model":"sculpt","settings":{"geometry_model":"base","texture_model":"baked"}}}'
};

try {
	const response = await fetch(url, options);
	const data = await response.json();
	console.log(data);
} catch (error) {
	console.error(error);
}
```

**Response Example**:
```json
{
  "_id": "SESSION_XXXXXXXXX_XXXXXXXX",
  "user_id": "XXXXXXXXXXXXXXXXXX",
  "status": "incomplete",
  "type": "image_to_kit",
  "input": {
    "image": {
      "_id": "ASSET_XXXXXXXXXX_XXXXXXX",
      "name": "",
      "status": "complete",
      "type": "image",
      "data": {
        "image_url": "https://example.com/image.jpg"
      }
    },
    "model": "sculpt",
    "settings": {
      "geometry_model": "base",
      "texture_model": "none",
      "topology": "tris",
      "resolution": 100000,
      "symmetry": "off",
      "scaled_bbox": [-1, -1, -1],
      "preserve_aspect_ratio": false,
      "pivot_point": [0, -0.5, 0]
    }
  },
  "output": {
    "part_images": [],
    "part_meshes": []
  }
}
```

### Chat-to-3D

Generate improved images through conversational prompts, then convert to 3D.

**Endpoint**: `POST https://api.csm.ai/v3/sessions`

**Request Body**:
```json
{
  "type": "chat_to_3d",
  "messages": [
    {
      "type": "user_prompt",
      "message": "...",
      "images": ["data:base64image", "https://...", "ASSET_xxxxxx"]
    }
  ]
}
```

**Response Example**:
```json
{
    "_id": "SESSION_1749141784_4235795",
    "user_id": "65dcd034c42248b5b1c48ddf",
    "status": "incomplete",
    "created_at": "2025-06-05T16:43:05.143Z",
    "updated_at": "2025-06-05T16:43:05.143Z",
    "type": "chat_to_3d",
    "messages": [
        {
            "_id": "CHAT_MSG_1749141785_5708367",
            "created_at": "2025-06-05T16:43:05.143Z",
            "message": "give me a 3/4 3d asset view of this",
            "type": "user_prompt",
            "context": null,
            "images": [
                {
                    "_id": "ASSET_1749141784_1360252",
                    "user_id": "65dcd034c42248b5b1c48ddf",
                    "name": "",
                    "parent_path": "/",
                    "status": "complete",
                    "jobs": [],
                    "created_at": "2025-06-05T16:43:05.076Z",
                    "updated_at": "2025-06-05T16:43:05.076Z",
                    "type": "image",
                    "data": {
                        "small_image_url": "",
                        "medium_image_url": "",
                        "image_url": "https://rawcapture.blob.core.windows.net/uploaded/ASSET_1749141784_1360252/input.png?sp=rcwl&st=2022-06-24T16:05:30Z&se=2025-07-17T00:05:30Z&spr=https&sv=2021-06-08&sr=c&sig=OQKXCAQ7akLUp%2BPxLdTplV3Bz0OTUadK9huuNe%2FJ3%2Fs%3D"
                    }
                }
            ]
        },
        {
            "_id": "CHAT_MSG_1749141785_1664595",
            "created_at": "2025-06-05T16:43:05.143Z",
            "type": "image_generation",
            "images": [
                {
                    "prompt": "give me a 3/4 3d asset view of this",
                    "asset": {
                        "_id": "ASSET_1749141785_6005361",
                        "user_id": "65dcd034c42248b5b1c48ddf",
                        "name": "",
                        "session_id": "SESSION_1749141784_4235795",
                        "parent_path": "/",
                        "status": "incomplete",
                        "jobs": [
                            {
                                "_id": "JOB_1749141785_7377593",
                                "display_type": "image_generation",
                                "status": "in_progress"
                            }
                        ],
                        "created_at": "2025-06-05T16:43:05.090Z",
                        "updated_at": "2025-06-05T16:43:05.090Z",
                        "type": "image",
                        "data": {
                            "small_image_url": "",
                            "medium_image_url": "",
                            "image_url": ""
                        }
                    }
                }
            ]
        }
    ]
}
```

**Note**: Wait for status to be `complete`, then use `messages[1].images[0].asset._id` as the `input.image` in subsequent Image-to-3D calls.

</details>
