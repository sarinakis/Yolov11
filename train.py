# Train args Documentation https://docs.ultralytics.com/modes/train/#train-settings

from ultralytics import YOLO

def main():
    # Model size e.g. yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt and yolo11x.pt
    # In order n->nano, s->small, m->medium, l->large and x->x-large
    model = YOLO('yolo11n.pt')

    train_results = model.train(
        data='custom_data.yaml',  # path to dataset YAML
        batch = 16, #with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
        epochs = 9999,  # number of training epochs
        imgsz = 640,  # training image size
        device = '0', # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        patience = 100, # Number of epochs to wait without improvement in validation metrics before early stopping the training.
        project = 'runs/train', # for ready to train code
        #project = 'runs/debug', # for debug
        name = 'exp',
        plots = True, #Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
        val = True, #Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.
        cls = 0.5, #Weight of the classification loss in the total loss function.
        save = True, # Enables saving of training checkpoints and final model weights.
        workers = 8, #Number of worker threads for data loading. Influences the speed of data preprocessing and feeding into the model.
        # We can add augmented values here as well
    )

if __name__ == "__main__":
    main()