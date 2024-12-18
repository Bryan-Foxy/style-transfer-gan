#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream> // For std::ofstream

namespace fs = std::filesystem;

// Supported video extensions
std::vector<std::string> video_extensions = {".mp4", ".mov", ".avi", ".mkv"};

// Check if the file has a supported video extension
bool is_supported_video(const std::string& filename) {
    for (const auto& ext : video_extensions) {
        if (filename.size() >= ext.size() &&
            filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

int main() {
    std::string input_dir = "results/inputs/";
    std::string output_dir = "results/outputs/";
    std::string final_output = "results/combined_video.mp4";

    // Collect matching input/output video pairs
    std::vector<std::pair<std::string, std::string>> video_pairs;

    // Iterate through files in the input directory
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && is_supported_video(entry.path().filename().string())) {
            std::string input_video = entry.path().string();
            std::string output_video = output_dir + entry.path().stem().string() + "_preds.mp4";

            // Check if the corresponding output video exists
            if (fs::exists(output_video)) {
                video_pairs.emplace_back(input_video, output_video);
            } else {
                std::cerr << "Warning: No matching output video found for " << input_video << std::endl;
            }
        }
    }

    // Check if there are any video pairs
    if (video_pairs.empty()) {
        std::cerr << "Error: No matching video pairs found in the specified directories." << std::endl;
        return 1;
    }

    // Temporary file for concatenated videos
    std::string temp_concat_file = "temp_concat_list.txt";
    std::ofstream concat_file(temp_concat_file);

    // Process each video pair
    for (size_t i = 0; i < video_pairs.size(); ++i) {
        const auto& [input_video, output_video] = video_pairs[i];
        std::string combined_output = "results/temp_combined_" + std::to_string(i) + ".mp4";

        // Wrap file paths in double quotes to handle spaces
        std::string ffmpeg_command = "ffmpeg -y -i \"" + input_video + "\" -i \"" + output_video +
                                     "\" -filter_complex \"[0:v]scale=256:256,fps=30[a];[1:v]scale=256:256,fps=30[b];[a][b]hstack=inputs=2\" -an \"" +
                                     combined_output + "\"";

        // Execute the ffmpeg command
        int result = system(ffmpeg_command.c_str());
        if (result != 0) {
            std::cerr << "Error: Unable to process video pair: " << input_video << " and " << output_video << std::endl;
            return 1;
        }

        // Write the combined video to the concatenation list
        concat_file << "file '" << combined_output << "'\n";
    }

    concat_file.close();

    // Concatenate all combined videos into one final output
    std::string concat_command = "ffmpeg -y -f concat -safe 0 -i \"" + temp_concat_file + "\" -c copy \"" + final_output + "\"";
    int concat_result = system(concat_command.c_str());

    // Clean up temporary files
    fs::remove(temp_concat_file);
    for (size_t i = 0; i < video_pairs.size(); ++i) {
        fs::remove("results/temp_combined_" + std::to_string(i) + ".mp4");
    }

    if (concat_result != 0) {
        std::cerr << "Error: Unable to concatenate combined videos." << std::endl;
        return 1;
    }

    std::cout << "Final combined video saved at: " << final_output << std::endl;
    return 0;
}