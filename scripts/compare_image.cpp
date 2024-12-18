#include <iostream>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    // Check if the image name is provided
    if (argc != 2) {
        std::cerr << "Usage: ./compare_images <image_name>" << std::endl;
        return 1;
    }

    // Get the image name from the arguments
    std::string image_name = argv[1];
    std::string input_image = "results/inputs/" + image_name;
    std::string output_image = "results/outputs/" + image_name.substr(0, image_name.find_last_of('.')) + "_preds.png";
    std::string combined_output = "results/" + image_name;

    // Construct the ffmpeg command
    std::string ffmpeg_command = "ffmpeg -y -i " + input_image +
                                 " -i " + output_image +
                                 " -filter_complex \"[0:v]scale=256:256[a];[1:v]scale=256:256[b];[a][b]hstack=inputs=2\" " +
                                 combined_output;

    // Execute the ffmpeg command
    int result = system(ffmpeg_command.c_str());

    // Check if the command executed successfully
    if (result != 0) {
        std::cerr << "Error: Unable to process images. Make sure ffmpeg is installed and the image paths are correct." << std::endl;
        return 1;
    }

    std::cout << "Combined image saved at: " << combined_output << std::endl;
    return 0;
}