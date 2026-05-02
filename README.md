# Moral Dilemma Simulation (MDS)

Welcome to the official repository for the paper: **"Visual Distraction Undermines Moral Reasoning in Vision-Language Models"**.

MDS is a dynamic, generative multimodal benchmark grounded in Moral Foundation Theory (MFT). It evaluates the moral reasoning of Vision-Language Models (VLMs) by identifying critical fragilities when processing the vision modality. Instead of a static dataset, MDS serves as a controllable generation engine that allows for the orthogonal manipulation of visual and contextual variables.

## 📂 Repository Structure

```text
MDS/
├── config/              # Configuration files for dilemmas and character attributes
├── baseline/            # Tri-modal evaluation scripts for SOTA VLMs
├── env                  # Pygame-based 2D sandbox visual rendering engine
├── data/                # Data generation pipeline for the three subsets
├── results/             # Evaluation results, analyze code and results
├── visualization/       # Visualization code and results
├── gradient_feature/    # Mechanistic analysis tools (Grad-CAM & Text Gradient)
└── style/               # Style-transfer ablation
```

## 🧩

### 1. `config/` (Dilemma settings)
This folder acts as the central nervous system of the generation pipeline.
* **`character.yaml`**: Defines the demographic and character variables used to test bias and social hierarchies (e.g., species, race, gender, profession, age).
* **`dilemma.yaml`**: Contains the structured textual templates for the 23 high-stakes moral dilemmas. It dictates how conceptual variables—such as *Personal Force* (direct vs. indirect harm), *Intention of Harm* (means vs. side-effect), and *Self-Benefit*—are toggled within the prompts.
* **`constants.py`**: Manages global constants and paths for the generation engine.

### 2. `baseline/` (Model evaluation)
Scripts to evaluate state-of-the-art models.
The scripts implement the **Tri-modal Diagnostic Protocol**:
1.  **Text Mode**: Tests upper-bound moral reasoning using structured text.
2.  **Caption Mode**: The model generates a caption and reads embedded text via OCR, evaluating reasoning based purely on informational complexity.
3.  **Image Mode**: Direct visual input, used to isolate the modality gap caused by visual distraction.

### 3. `env/gui/` (Sandbox rendering engine)
The visual rendering system generates pixel-art sandbox scenes that faithfully depict dilemma scenarios while minimizing artistic confounds. 
* **`canvas.py` & `cell.py`**: Handle the spatial layout, background mapping (e.g., Train, School, Road, Hospital), and coordinate calculations for the scenes.
* **`character.py`**: Translates the logical character attributes from `character.yaml` into specific visual avatars and positions them on the grid.

### 4. `data/` (Dataset generation)
Contains the scripts to programmatically generate the three diagnostic subsets described in the paper:
* **`quantity/generate.py`**: Generates scenarios testing *utilitarian sensitivity* by altering the ratio of lives saved vs. sacrificed, keeping visual demographic attributes neutral.
* **`single_feature/generate.py`**: Isolates specific demographic or conceptual variables (varying one feature at a time) to trace individual drivers of moral bias.
* **`interaction/generate.py`**: Generates high-dimensional intersectional scenarios (e.g., simultaneously manipulating quantity, race, and profession) to reveal complex, combinatorial biases.

### 5. `results/` (Analysis scripts and results)
* **`results/xxx`**: Model evaluation results, analytic scripts and analysis results of the xxx subset.

### 6. `visualization/` (Visualization scripts and results)
* **`visualization/xxx`**: Visualization scripts and results of the xxx subset.

### 7. `gradient_feature/` (Mechanistic analysis)
Contains tools and results of gradient-based feature attribution.
* **`extract_text_gradient.py` & `plot_text_gradient.py`**: Uses Gradient x Input to measure token saliency in Text/Caption modes.
* **`extract_gradcam.py` & `plot_gradcam_map.py`**: Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the spatial distribution of the model's attention in Image Mode, highlighting the noisy, distracted saliency.

### 8. `style/` (Style ablation)
Contains the examples for the Style-transfer ablation study to prove the visual distraction effect is not tied to the sandbox aesthetic.
* **`style/xxx`**: Raw and 9 style transferred images of the dilemma xxx.

### 9. questionnaire.pdf
The questionnaire for human visual identification study.

---

## 🛠️ Extensibility: How to add new tasks

A core contribution of MDS is that it is a highly extensible, programmatic generation engine rather than a static dataset. You can easily expand the benchmark to include novel moral conflicts or specialized edge cases with minimal overhead.

To introduce a new task or scenario, follow these steps:

### Step 1: Define the textual template (`config/dilemma.yaml`)
Add your new scenario to the configuration file. You must provide a structured description featuring placeholder variables (e.g., `{agent_color}`, `{rail_vertical_quantity}`) that represent conceptual and character attributes. 
For example, to configure personal force, define how the template changes when the variable is `0` (absent) versus `1` (present).

### Step 2: Configure the visual scene
You do not need complex rendering expertise. You only need to provide a basic scene configuration that maps your textual variables to the engine's built-in pixel-art assets. 

Since the visual scenes are rendered using licensed assets, they cannot be publicly released. If there is interest in extending the dataset with additional scenarios, please feel free to contact us to render new scenarios.

### Step 3: Run the Generation Engine (`data/`)
Once the template and layout are defined, simply use the scripts in the `data/` directory. The MDS engine will automatically handle:
1. Orthogonal manipulation of your variables.
2. Prompt rewriting for fluency via the GPT-4 pipeline.
3. The multi-modal rendering of the image and the generation of the corresponding ground-truth configuration file.

You can generate new data following the provided data generation scripts, adapting them to your desired experimental design.