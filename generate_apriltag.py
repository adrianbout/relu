import cv2
import numpy as np
import apriltag

# INSTALL BEFORE apriltag ON CMD
# winget install Kitware.CMake
# winget install Microsoft.VisualStudio.2022.BuildTools



def generate_apriltag(tag_id=0, tag_family="tag36h11", size=200, output_file="apriltag.png"):
    """
    Generates an AprilTag and saves it as an image.

    :param tag_id: ID of the tag to generate
    :param tag_family: Family of AprilTags (default: "tag36h11")
    :param size: Size of the generated image in pixels
    :param output_file: Output filename to save the image
    """
    # Create detector
    detector = apriltag.Detector(families=tag_family)

    # Generate the tag
    tag = detector.tags[tag_family].generator(tag_id)

    # Resize for better visibility
    tag = cv2.resize(tag.astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST)

    # Save to file
    cv2.imwrite(output_file, tag)
    print(f"AprilTag saved as {output_file}")

# Generate an AprilTag with ID 0
generate_apriltag()
