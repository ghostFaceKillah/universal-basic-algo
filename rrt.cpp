#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <tuple>
#include <vector>


class System {
public:
    int state_size() {
        return 2;
    }

    int control_size() {
        return 2;
    }

    // Returns new state
    Eigen::RowVectorXd drive_toward(
        Eigen::RowVectorXd start_state,
        Eigen::RowVectorXd target_state
    ) {
        return (target_state - start_state).normalized() / 2.0 + start_state;
    }
};


class Rng {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<> unit_uniform;

public:
    Rng() : gen(std::random_device{}()), unit_uniform(0.0, 1.0) {
    }

    Eigen::RowVectorXd get_random_vector(float low = 0., float high = 1.) {
        Eigen::RowVectorXd sample(2);
        sample(0) = unit_uniform(gen);
        sample(1) = unit_uniform(gen);

        return sample * (high - low) + Eigen::RowVectorXd::Constant(2, low);
    }

    /*
    *
    auto rand_normal = [&](double mean, double std) -> double { return std * dis(gen) + mean; };
    auto unit_normal = [&]() -> double { return rand_normal(0, 1); };
     */
};

class CollisionResolver {
public:
    float robot_radius;

    // circular obstacles - the dimensions are x, y, radius
    std::vector<std::tuple<Eigen::RowVectorXd, float> > obstacles;

    CollisionResolver() {
        robot_radius = 0.1;
        Eigen::RowVectorXd obstacle_1(2);
        obstacle_1 << 0., 2.;

        Eigen::RowVectorXd obstacle_near_origin(2);
        obstacle_near_origin << 1.0, 1.0;
        float radius_near_origin = 0.5;

        Eigen::RowVectorXd large_obstacle(2);
        large_obstacle << -2.0, 3.0;
        float large_obstacle_radius = 1.5;

        Eigen::RowVectorXd distant_obstacle(2);
        distant_obstacle << 5.0, -4.0;
        float distant_obstacle_radius = 1.0;

        Eigen::RowVectorXd small_obstacle1(2), small_obstacle2(2), small_obstacle3(2);
        small_obstacle1 << -1.0, -1.0; // Obstacle 1
        small_obstacle2 << -1.5, -1.2; // Obstacle 2
        small_obstacle3 << -1.2, -1.5; // Obstacle 3
        float small_obstacle_radius = 0.3;


        obstacles = {
            {obstacle_1, 0.71},
        };

        obstacles.push_back({obstacle_near_origin, radius_near_origin});
        obstacles.push_back({large_obstacle, large_obstacle_radius});

        obstacles.push_back({distant_obstacle, distant_obstacle_radius});

        obstacles.push_back({small_obstacle1, small_obstacle_radius});
        obstacles.push_back({small_obstacle2, small_obstacle_radius});
        obstacles.push_back({small_obstacle3, small_obstacle_radius});
    }

    bool has_collided(
        Eigen::RowVectorXd start_state,
        Eigen::RowVectorXd end_state
    ) {
        // very naive, don't care for now

        // compute distances between
        for (const auto &obstacle: obstacles) {
            Eigen::RowVectorXd obstacle_center = std::get<0>(obstacle);
            float obstacle_radius = std::get<1>(obstacle);

            if ((end_state - obstacle_center).norm() < obstacle_radius + robot_radius) {
                return true;
            }
        }

        return false;
    }
};


// Function to find the closest vector
int find_closest_idx(const Eigen::MatrixXd &matrix, const Eigen::RowVectorXd &a, int filled_in_until) {
    int closestRow = -1;
    float minDistance = std::numeric_limits<float>::max();


    for (int i = 0; i <= filled_in_until; ++i) {
        Eigen::RowVectorXd diff = matrix.row(i) - a;
        float distance = diff.norm();
        if (distance < minDistance) {
            minDistance = distance;
            closestRow = i;
        }
    }

    return closestRow;
}

void visualize(const std::vector<std::tuple<Eigen::RowVectorXd, float> > &obstacles,
               const Eigen::MatrixXd &data,
               const std::vector<int> &came_from,
               int filled_in_until,
               const Eigen::RowVectorXd &candidate_state,
               const Eigen::RowVectorXd &sample_state,
               const Eigen::RowVectorXd &goal) {
    // Create an image for drawing
    int width = 600, height = 600;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // Offset to translate (0, 0) to the center of the image
    int offsetX = width / 2;
    int offsetY = height / 2;

    // Scale factor to fit the points within the image dimensions
    float scale = 40.0;

    // Draw obstacles
    for (const auto &obstacle: obstacles) {
        Eigen::RowVectorXd center = std::get<0>(obstacle);
        float radius = std::get<1>(obstacle);

        cv::Point center_cv(offsetX + center[0] * scale, offsetY - center[1] * scale);
        cv::circle(image, center_cv, static_cast<int>(radius * scale), cv::Scalar(255, 0, 0), -1);
    }

    // Draw states and connections
    for (int i = 0; i <= filled_in_until; ++i) {
        Eigen::RowVectorXd state = data.row(i);
        cv::Point state_cv(offsetX + state[0] * scale, offsetY - state[1] * scale);

        // Draw state
        cv::circle(image, state_cv, 3, cv::Scalar(0, 255, 0), -1);

        // Draw line to the state it came from
        if (i > 0) {
            Eigen::RowVectorXd parent_state = data.row(came_from[i]);
            cv::Point parent_state_cv(offsetX + parent_state[0] * scale, offsetY - parent_state[1] * scale);
            cv::line(image, state_cv, parent_state_cv, cv::Scalar(0, 255, 255), 1);
        }
    }

    // Draw candidate_state
    cv::Point candidate_cv(offsetX + candidate_state[0] * scale, offsetY - candidate_state[1] * scale);
    cv::circle(image, candidate_cv, 4, cv::Scalar(0, 0, 255), -1); // Blue

    // Draw sample_state
    cv::Point sample_cv(offsetX + sample_state[0] * scale, offsetY - sample_state[1] * scale);
    cv::circle(image, sample_cv, 4, cv::Scalar(255, 255, 0), -1); // Cyan

    // Draw goal
    cv::Point goal_cv(offsetX + goal[0] * scale, offsetY - goal[1] * scale);
    cv::circle(image, goal_cv, 5, cv::Scalar(0, 255, 255), -1); // Yellow


    // Display the image
    cv::imshow("Environment", image);
    cv::waitKey(0); // Wait for a keystroke in the window
}


void run_rrt() {
    System system;
    Rng random;
    CollisionResolver collision_resolver;

    int graph_size = 512;
    int filled_in_until = 0; // inclusive
    int closest_idx;

    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(graph_size, system.state_size());
    std::vector<int> came_from(graph_size, 0);
    data.row(0) = Eigen::RowVectorXd::Zero(2);

    Eigen::RowVectorXd rollout_state(2);
    Eigen::RowVectorXd candidate_state(2);
    Eigen::RowVectorXd sample_state(2);
    Eigen::RowVectorXd goal(2);
    goal << 0., 4.;

    for (int i = 0; i < 1000; ++i) {
        // draw a random point B in the space
        sample_state = random.get_random_vector(-6., 6.);

        // take  point A from the datastructure closest to this point
        closest_idx = find_closest_idx(data, sample_state, filled_in_until);
        rollout_state = data.row(closest_idx);

        std::cout << "Went from " << rollout_state << std::endl;

        // drive from A to B - maybe randomly, maybe using drive function
        candidate_state = system.drive_toward(rollout_state, sample_state);
        std::cout << "     to   " << candidate_state << std::endl;

        // check if we collided
        bool collided = collision_resolver.has_collided(rollout_state, candidate_state);

        if (collided) {
            std::cout << "     collided!" << std::endl;
        } else {
            std::cout << "     didn't collide!" << std::endl;
        }

        // if didn't colide, add to the tree
        //    matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange); if needed
        if (!collided) {
            if (filled_in_until + 1 == graph_size) {
                // Double the number of rows when resizing
                graph_size *= 2;
                data.conservativeResize(graph_size, Eigen::NoChange);
                came_from.resize(graph_size);
            }

            data.row(filled_in_until + 1) = candidate_state;
            came_from[filled_in_until + 1] = closest_idx;
            filled_in_until++;
        }

        // draw the state
        // the code that you will write will be here
        visualize(collision_resolver.obstacles, data, came_from, filled_in_until, candidate_state, sample_state, goal);
    }
}

int main() {
    // example_drawing_code();
    run_rrt();
}
