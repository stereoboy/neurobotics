#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include <std_msgs/Int8.h>
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/OccupancyGrid.h"
#include <tf/transform_datatypes.h>

#include <iostream>
#include<vector>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

bool yaw_constraint_flag = false;

// Publisher and subscribers
bool initialized = false;
ros::Publisher stage_pub;

actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> *ac_;

// Uncomment when using real amcl localization
// ros::Publisher move_base_pose_pub;
int sampleArea = 1;
unsigned int pixel_position;
bool costmap_there = false;

double margin = 0.5;
double x_max = 19.5;
double x_min = 0.5;
double y_max = 19.5;
double y_min = 0.5;
double yaw_max = 3.10;
double yaw_min = -3.10;
double o = 0.0;
double resolution = 0.05; // Resolution of the map, meters/pixel. Check *.yaml
int threshold_pose_valid = 0;
int map_size = 400;

double stddev = 4.0;

double new_pose_x = 0.0;
double new_pose_y = 0.0;

// Seed the random numbers
boost::mt19937 rng(42);

std::vector<nav_msgs::Odometry> robot_poses;

nav_msgs::OccupancyGrid current_costmap;


double getRandomDouble(double min, double max, double offset)
{
    int range = int((max - min) * 100);
    return (double)(rand() % range)/100 + min + offset;
}

double dist(double x_1, double y_1, double x_2, double y_2)
{
    return sqrt(pow((x_1 - x_2), 2.0) + pow((y_1 - y_2), 2.0));
}

void publishNewGoal()
{
    double x;
    double y;
#
    // Initialize the random value
    boost::normal_distribution<> nd(0.0, stddev);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    // Check if the point is unoccupied
    bool collision = true;
    while (collision)
    {
        collision = false;
        x = new_pose_x + var_nor();
        y = new_pose_y + var_nor();

        // First check for the max x and y values
        if (x > x_max || x < x_min || y > y_max || y < y_min)
        {
            collision = true;
        }
        else
        {
            // Now check for the costmap values
            pixel_position = (unsigned int)(x / resolution) + (unsigned int)(y / resolution) * map_size;
            if (costmap_there && current_costmap.data.at(pixel_position) > threshold_pose_valid)
            {
                collision = true;
                ROS_ERROR("NewGoal() failed due to collision with costmap!");
            }
            else
            {
//                // Now check for dynamic obstacles
//                for (long unsigned i = 0; i < robot_poses.size(); i++)
//                {
//                    if (dist(x, y, robot_poses.at(i).pose.pose.position.x, robot_poses.at(i).pose.pose.position.y) < 0.8)
//                    {
//                        collision = true;
//                        ROS_ERROR("NewGoal() failed due to moving obastcles!");
//                    }
//                }
            }
        }
    }
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.z = 0.0;
    pose_stamped.pose.position.x = x;
    pose_stamped.pose.position.y = y;
    pose_stamped.pose.orientation.z = 1.0;
    pose_stamped.pose.orientation.w = o;
    pose_stamped.header.frame_id = "map";
    pose_stamped.header.stamp = ros::Time::now();
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose = pose_stamped;
    ac_->sendGoal(goal);
}

void publishNewPose()
{

    double x;
    double y;
    double yaw;

    // Check if the point is unoccupied
    bool collision = true;
    while (collision)
    {
        collision = false;
        x = getRandomDouble(x_min, x_max, 0.00);
        y = getRandomDouble(y_min, y_max, 0.00);

        yaw = getRandomDouble(yaw_min, yaw_max, 0.00);

        ROS_ERROR("NewPose() (%f, %f, %f)", x, y, yaw);

        // First check for the costmap values
        pixel_position = (unsigned int)(x / resolution) + (unsigned int)(y / resolution) * map_size;
        if (costmap_there && current_costmap.data.at(pixel_position) > threshold_pose_valid)
        {
            collision = true;
            ROS_ERROR("NewPose() failed due to collision with costmap!");
        }
        else
        {
            for (long unsigned i = 0; i < robot_poses.size(); i++)
            {
                // Now check for dynamic obstacles
                if (dist(x, y, robot_poses.at(i).pose.pose.position.x, robot_poses.at(i).pose.pose.position.y) < 0.8)
                {
                    collision = true;
                    ROS_ERROR("NewPose() failed due to moving obastcles!");
                }
            }
        }
    }

    geometry_msgs::Pose pose;
    pose.position.z = 0.0;
    pose.position.x = new_pose_x = x;
    pose.position.y = new_pose_y = y;
    if (yaw_constraint_flag) {
        pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, yaw); // http://docs.ros.org/api/tf/html/c++/transform__datatypes_8h.html
    }
    else {
        pose.orientation.z = 1.0;
        pose.orientation.w = o;
    }
    stage_pub.publish(pose);
}

void botCallback(const std_msgs::Bool new_round)
{
    if(new_round.data)
    {
        // Send new position to stage
        publishNewPose();

        // Make sure that the global planner is aware of the new position
        ros::Rate r(4);
        r.sleep();

        // Send new goal position to move_base
        publishNewGoal();
    }
}

void newSampleAreaCallback(const std_msgs::Int8 newSampleAreaMsg)
{
    sampleArea = newSampleAreaMsg.data;
    switch (sampleArea)
    {
        case 1:
            x_max = 1.20;
            x_min = -1.40;
            y_max = 3.40;
            y_min = 0.80;
            break;
        case 2:
            x_max = 1.20;
            x_min = -1.40;
            y_max = 3.40;
            y_min = -1.50;
            break;
        case 3:
            x_max = 3.00;
            x_min = -1.40;
            y_max = 3.40;
            y_min = -1.50;
            break;
        case 4:
            x_max = 5.00;
            x_min = -1.40;
            y_max = 3.40;
            y_min = -1.50;
            break;
        default:
            1;
    }
}

#define DEFINE_ROBOT_CALLBACK(id)  \
    void robot_##id##_callback(nav_msgs::Odometry msg) \
    { \
        ROS_INFO(">>>%s(%d)", __FUNCTION__, id); \
        robot_poses.at(id - 1) = msg; \
        ROS_INFO("<<<%s(%d)", __FUNCTION__, id); \
    }\

#define REGISTER_ROBOT_CALLBACK(node, id) \
    ROS_INFO("(%d)", id); \
    ros::Subscriber sub_robot_##id = node.subscribe("/robot_" #id "/base_pose_ground_truth", 1000, robot_##id##_callback); \
    ROS_INFO("/robot_" #id "/base_pose_ground_truth"); \

DEFINE_ROBOT_CALLBACK(1)
DEFINE_ROBOT_CALLBACK(2)
DEFINE_ROBOT_CALLBACK(3)
DEFINE_ROBOT_CALLBACK(4)

//void robot_1_callback(nav_msgs::Odometry msg)
//{
//    robot_poses.at(0) = msg;
//}
//
//void robot_2_callback(nav_msgs::Odometry msg)
//{
//    robot_poses.at(1) = msg;
//}
//

void costmapCallback(nav_msgs::OccupancyGrid msg)
{
    costmap_there = true;
    current_costmap = msg;
    resolution = msg.info.resolution;
    map_size = msg.info.width;
    x_max = msg.info.width*resolution - margin;
    x_min = margin;
    y_max = msg.info.height*resolution;
    y_min = margin;
    ROS_ERROR("costmap: resolution: %f, dim: (%dx%d)", msg.info.resolution, msg.info.height, msg.info.width);

    initialized = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "neuro_training_bot");

    ros::NodeHandle n;

    // Parameters
    std::string robot_type_str;
    n.param("/robot_type", robot_type_str, std::string("holonomic"));
    //yaw_constraint_flag = (robot_type_str.compare(std::string("nonholonomic")) == 0);
    //yaw_constraint_flag = false;
    yaw_constraint_flag = true;

    // Subscribers
    //ros::Subscriber sub_planner = n.subscribe("/move_base/NeuroLocalPlannerWrapper/new_round", 1000, botCallback);
    ros::Subscriber sub_recovery = n.subscribe("/move_base/neuro_fake_recovery/new_round", 1000, botCallback);
    ros::Subscriber sub_area = n.subscribe("/sampleArea", 1000, newSampleAreaCallback);
    ros::Subscriber sub_costmap = n.subscribe("/move_base/global_costmap/costmap", 1000, costmapCallback);

//    REGISTER_ROBOT_CALLBACK(n, 1);
//    REGISTER_ROBOT_CALLBACK(n, 2);
//    REGISTER_ROBOT_CALLBACK(n, 3);
//    REGISTER_ROBOT_CALLBACK(n, 4);

    // Publishers
    stage_pub = n.advertise<geometry_msgs::Pose>("neuro_stage_ros/set_pose", 1);

    // Uncomment when using real amcl localization
    //move_base_pose_pub = n.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1);
    ros::Duration time_between_ros_wakeups(0.2);
    while (!initialized || sub_costmap.getNumPublishers() < 1)
    {
        ROS_WARN("Waiting for global_costmap");
        ros::spinOnce();
        time_between_ros_wakeups.sleep();
    }

    // Make sure that the global planner is aware of the new position
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac("/move_base", true);
    ac_ = &ac;
    ac.waitForServer(); //will wait for infinite time

    std::vector<double> initial_positions;
    while (!n.getParam("initial_positions", initial_positions) || !initial_positions.size())
    {
        ROS_WARN("Waiting for initial_positions");
        ros::Rate r(4);
        r.sleep();
    }

    for(int i = 0; i < initial_positions.size()/2; i++)
    {
        nav_msgs::Odometry temp_pose;
        temp_pose.pose.pose.position.x = initial_positions[2*i];
        temp_pose.pose.pose.position.y = initial_positions[2*i + 1];
        temp_pose.pose.pose.position.z = 0.0;
        temp_pose.pose.pose.orientation.z = 1.0;
        robot_poses.push_back(temp_pose);
        ROS_INFO("--------------------->[%d] (%f, %f)", i, initial_positions[2*i], initial_positions[2*i + 1]);
    }

    ROS_INFO("neuro_training_bot: Initialization done");

    while (ros::ok()) {
        ros::spinOnce();

        // Send new position to stage
        publishNewPose();

        // Make sure that the global planner is aware of the new position
        ros::Rate r(4);
        r.sleep();

        // Send new goal position to move_base
        publishNewGoal();

        ac.waitForResult(ros::Duration(200.0));

        if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            ROS_INFO("Excellent! Your robot has reached the goal position.");
        else
            ROS_INFO("The robot failed to reach the goal position");
    }

    return 0;
}
