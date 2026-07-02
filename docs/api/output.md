# Saving results

The output can be stored either in a single JSON file or in multiple JSON files. Each JSON file contains an array of objects.

## Output
The `save_json` function is responsible for saving the results. A dedicated directory named `results_yyyymmdd_hhmmss` is automatically created to store all generated files.
::: scentree.io.writer.save_json
    options:
      heading_level: 3
      members: true

## Description of the file
The JSON file contains an array of objects. Each object includes the following fields:

| Field | Type | Description |
|------|------|-------------|
| `num_scenarios` | `integer` | Number of generated scenarios. |
| `num_stages` | `integer` | Number of stages in the stochastic problem. |
| `in_sample_prediction` | `boolean` | Indicates whether the generated scenarios correspond to observed days (`true`) or future days (`false`). |
| `scenario_tree_data` | `array` of `array` of `float` | Scenario data. It contains one array per scenario, where each inner array represents a single scenario. |
| `mean_value_scenario_tree` | `array` of `float` | Mean value of the generated scenarios. |
| `predicted_value` | `array` of `float` | Predicted values. |
| `observed_value` | `array` of `float` or `null` | Observed values. If `in_sample_prediction` is `true`, this field contains an array of length `data_dim`; otherwise, it is `null`. |
| `scenario_probabilities` | `array` of `float` | Probability associated with each generated scenario. |
| `mapping_datasets_columns` | `array` of `object` | Mapping between the original datasets and the corresponding columns in the scenario data. Each object contains the dataset name and the associated column indices. |
| `tree` | `array` of `object` | Description of the generated scenario tree. The structure of each object is described in the table below. |

### Tree object
| Field | Type | Description |
|------|------|-------------|
| `key` | `int[2]` | A pair representing `(stage, position in stage)`. |
| `scenario_ids` | `int[]` | Array of scenario IDs that belong to the same cluster. |
| `parent_key` | `int[2]` or `null` | Parent key of the current node. It is `null` for root nodes. |
| `description` | `string` | Textual description of the `key` field. |
