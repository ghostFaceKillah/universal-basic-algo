#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <cmath>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>

void printMaxPerDimension(const std::set<std::pair<int, int> > &open_set,
                          const std::set<std::pair<int, int> > &closed_set) {
    if (open_set.empty() && closed_set.empty()) {
        std::cout << "Both sets are empty." << std::endl;
        return;
    }

    int maxXOpen = INT_MIN, maxYOpen = INT_MIN;
    int maxXClosed = INT_MIN, maxYClosed = INT_MIN;

    // Find max in open_set
    for (const auto &elem: open_set) {
        maxXOpen = std::max(maxXOpen, elem.first);
        maxYOpen = std::max(maxYOpen, elem.second);
    }

    // Find max in closed_set
    for (const auto &elem: closed_set) {
        maxXClosed = std::max(maxXClosed, elem.first);
        maxYClosed = std::max(maxYClosed, elem.second);
    }

    std::cout << "Open Set Max X: " << maxXOpen << ", Max Y: " << maxYOpen << std::endl;
    std::cout << "Closed Set Max X: " << maxXClosed << ", Max Y: " << maxYClosed << std::endl << std::endl;
}

void visualize_astar(
    const std::vector<std::vector<bool> > &grid,
    const std::set<std::pair<int, int> > &open_set,
    const std::set<std::pair<int, int> > &closed_set,
    const std::pair<int, int> &start,
    const std::pair<int, int> &goal,
    int cell_size = 6
) {
    int height = grid.size();
    int width = grid[0].size();

    printMaxPerDimension(open_set, closed_set);

    // Create an image with 3 channels (RGB)
    cv::Mat image(height * cell_size, width * cell_size, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the grid
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (grid[i][j]) {
                // Draw obstacle
                cv::rectangle(image,
                              cv::Point(j * cell_size, i * cell_size),
                              cv::Point((j + 1) * cell_size, (i + 1) * cell_size),
                              cv::Scalar(0, 0, 0),
                              cv::FILLED);
            }
        }
    }

    // Draw the open set
    for (const auto &cell: open_set) {
        cv::rectangle(image,
                      cv::Point(cell.second * cell_size, cell.first * cell_size),
                      cv::Point((cell.second + 1) * cell_size, (cell.first + 1) * cell_size),
                      cv::Scalar(0, 255, 0),
                      cv::FILLED);
    }

    // Draw the closed set
    for (const auto &cell: closed_set) {
        cv::rectangle(image,
                      cv::Point(cell.second * cell_size, cell.first * cell_size),
                      cv::Point((cell.second + 1) * cell_size, (cell.first + 1) * cell_size),
                      cv::Scalar(0, 0, 255),
                      cv::FILLED);
    }

    // Draw start and goal
    cv::circle(image,
               cv::Point(start.second * cell_size + cell_size / 2, start.first * cell_size + cell_size / 2),
               cell_size / 4,
               cv::Scalar(0, 255, 0),
               cv::FILLED);
    cv::circle(image,
               cv::Point(goal.second * cell_size + cell_size / 2, goal.first * cell_size + cell_size / 2),
               cell_size / 4,
               cv::Scalar(0, 0, 255),
               cv::FILLED);

    // Display the image
    cv::imshow("a_star", image);
    cv::waitKey(-1); // Update the window and wait for a short period
}

std::vector<std::pair<int, int> > get_successors(const std::pair<int, int> &cur,
                                                 const std::vector<std::vector<bool> > &grid) {
    std::vector<std::pair<int, int> > successors;
    const int dx[4] = {1, 0, -1, 0};
    const int dy[4] = {0, 1, 0, -1};

    for (int i = 0; i < 4; ++i) {
        int newX = cur.first + dx[i];
        int newY = cur.second + dy[i];

        if (newX >= 0 && newX < grid.size() && newY >= 0 && newY < grid[0].size() && !grid[newX][newY]) {
            successors.push_back({newX, newY});
        }
    }

    return successors;
}

int heuristic(int x, int y, int goal_x, int goal_y) {
    return abs(x - goal_x) + abs(y - goal_y);
    //return std::sqrt(std::pow(x - goal_x, 2) + std::pow(y - goal_y, 2));

    // return std::min(abs(x - goal_x), abs(y - goal_y));
    // return std::max(abs(x - goal_x), abs(y - goal_y));
}


std::vector<std::pair<int, int> > a_star(
    std::vector<std::vector<bool> > grid,
    std::pair<int, int> start,
    std::pair<int, int> goal,
    bool verbose = false
) {
    if (verbose) {
        std::cout << "goal is " << goal.first << "," << goal.second << std::endl;
    }
    std::vector<std::vector<int> > f(grid.size(), std::vector<int>(grid[0].size(), 0));
    f[start.first][start.second] = 0;
    std::vector<std::vector<int> > g(grid.size(), std::vector<int>(grid[0].size(), std::numeric_limits<int>::max()));
    g[start.first][start.second] = 0;

    std::map<std::pair<int, int>, std::pair<int, int> > came_from;

    auto comp = [&f](const std::pair<int, int> &a, const std::pair<int, int> &b) {
        return f[a.first][a.second] > f[b.first][b.second];
    };

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >, decltype(comp)> q(comp);
    std::set<std::pair<int, int> > closed_set;
    std::set<std::pair<int, int> > open_set;

    open_set.insert(start);
    q.push(start);

    int i = 0;

    while (!q.empty()) {
        std::pair<int, int> cur = q.top();
        q.pop();

        if (verbose) {
            std::cout << "-------------------------" << std::endl;
            std::cout << "closed set = " << std::endl;
            for (const auto &m: closed_set) {
                std::cout << "(" << m.first << "," << m.second << ") ";
            }
            std::cout << std::endl;

            std::cout << "open set =" << std::endl;
            for (const auto &m: open_set) {
                std::cout << "(" << m.first << "," << m.second << ",f=" << f[m.first][m.second] << ") ";
            }
            std::cout << std::endl;


            std::cout << "popped " << cur.first << " " << cur.second << std::endl
                    << "   with f = " << f[cur.first][cur.second] << std::endl
                    << "        g = " << g[cur.first][cur.second] << std::endl
                    << "        h = " << heuristic(cur.first, cur.second, goal.first, goal.second) << std::endl;
        }


        if (cur == goal) {
            if (verbose) {
                std::cout << "found goal!" << std::endl;
            }
            std::vector<std::pair<int, int> > path;
            while (cur != start) {
                path.push_back(cur);
                cur = came_from[cur];
            }
            reverse(path.begin(), path.end());
            return path;
        }

        if (closed_set.find(cur) != closed_set.end()) {
            continue;
        }

        open_set.erase(cur);

        // add to closed set
        closed_set.insert(cur);


        for (const std::pair<int, int> &next: get_successors(cur, grid)) {
            // closed set
            if (closed_set.find(next) != closed_set.end()) {
                continue;
            }

            // open set
            bool inserted_into_open_set = false;

            if (open_set.find(next) == open_set.end()) {
                open_set.insert(next);
                inserted_into_open_set = true;
            }

            int g_hat = g[cur.first][cur.second] + 1;

            // update g perhaps
            if (inserted_into_open_set || g_hat < g[next.first][next.second]) {
                g[next.first][next.second] = g_hat;
                int h = heuristic(next.first, next.second, goal.first, goal.second);
                //f[next.first][next.second] = g_hat + h;
                f[next.first][next.second] = h;
                came_from[next] = cur;
                // we may consider next many times
                q.emplace(next);

                if (verbose) {
                    std::cout << "inserting (" << next.first << "," << next.second << ")" << std::endl
                            << "   with f = " << f[next.first][next.second] << std::endl
                            << "        g = " << g[next.first][next.second] << std::endl
                            << "        h = " << h << std::endl;
                }
            }
        }
        // code that visualizes things will be here
        if (i++ % 10 == 0) {
            visualize_astar(grid, open_set, closed_set, start, goal);
        }

        if (verbose) {
            std::cout << std::endl;
        }
    }

    return {};
}

std::vector<std::vector<bool> > initialize_random_grid(int height, int width, double obstacleProbability = 0.3) {
    std::vector<std::vector<bool> > grid(height, std::vector<bool>(width, false));

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Generate a random number and determine if the cell should be an obstacle
            grid[i][j] = (static_cast<double>(std::rand()) / RAND_MAX) < obstacleProbability;
        }
    }

    return grid;
}

int main() {
    // int rows = 5, cols = 5;
    // std::vector<std::vector<bool> > grid(rows, std::vector<bool>(cols, false));
    int height = 150;
    int width = 200;
    std::vector<std::vector<bool> > grid = initialize_random_grid(height, width);
    grid[0][0] = false;
    grid[height - 1][width - 1] = false;

    std::vector<std::pair<int, int> > path = a_star(grid, {0, 0}, {height - 1, width - 1});

    for (const auto &p: path) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }

    std::cout << std::endl;

    return 0;
}
