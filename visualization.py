import matplotlib.pyplot as plt
from PIL import Image


def prepprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def visualize_prediction(orginal_image, probabilities, class_name):
    fig, axrr = plt.subplots(1, 2, figsize=(14, 7))

    axrr[0].imshow(orginal_image)
    axrr[0].axis("off")

    axrr[1].barh(class_name, probabilities)
    axrr[1].set_xlabel("Probability")
    axrr[1].set_title("Class predictions")
    axrr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
