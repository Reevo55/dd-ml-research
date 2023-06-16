# Fake news detection methods research and proposal for M3FENDv2

## Description

This is a modified version of the [ICTMCG/M3FEND](https://github.com/ICTMCG/M3FEND) repository for Multimodal Multitask Fake News Detection. The key modifications are the integration with PyTorch Lightning, a lightweight PyTorch wrapper that helps to organize PyTorch code, and TensorBoard, a tool for providing the measurements and visualizations needed during the machine learning workflow.

## Key Features

- Compatibility with TensorBoard for visualizing the performance of the model and identifying bottlenecks in the code.
- Utilizes the simplicity and flexibility of PyTorch Lightning to organize PyTorch code and easily run it on CPUs, GPUs or TPUs.
- Custom data loader for efficient data processing and preparation.
- Extensive configurability for hyperparameters and training setup.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- HuggingFace Transformers

## Usage

After setting up the project, you can run the model training with the following command:

```sh
python main.py
```

## Output

The model parameters are saved in the `./params` directory and the TensorBoard logs are saved in the `./logs/my_experiment/M3FEND` directory. The model's performance on the test set is printed at the end of the script.

## Customization

You can modify this script to suit your needs. Some common modifications might include:

- Adjusting the model's hyperparameters.
- Adding additional callbacks or changing the logging settings.
- Modifying the data loading and processing logic.
- Changing the model architecture or training strategy.

Remember to also update the `ModelFactory` and `MyDataloader` classes to suit your new configuration if necessary.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
