#ifndef NEURO_LOCAL_PLANNER_WRAPPER_NEURO_LOCAL_PLANNER_WRAPPER_H_
#define NEURO_LOCAL_PLANNER_WRAPPER_NEURO_LOCAL_PLANNER_WRAPPER_H_

#include <tf/transform_listener.h>
#include <angles/angles.h>
#include <nav_msgs/Odometry.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <nav_core/base_local_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <pluginlib/class_loader.h>

#include <neuro_local_planner_wrapper/Transition.h>

#include <nav_msgs/OccupancyGrid.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <sensor_msgs/LaserScan.h>

#include <visualization_msgs/MarkerArray.h> // to_delete

#include <tf/tf.h>

#include <fstream>

// We use namespaces to keep things seperate under all the planners
namespace neuro_local_planner_wrapper
{
    class NeuroLocalPlannerWrapper : public nav_core::BaseLocalPlanner
    {
        public:

            // Constructor
            NeuroLocalPlannerWrapper();

            // Desctructor
            ~NeuroLocalPlannerWrapper();

            // Initialize the planner
            void initialize(std::string name, tf::TransformListener* tf, costmap_2d::Costmap2DROS* costmap_ros);

            // Sets the plan
            bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan);

            // Compute the velocity commands
            bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

            // Tell if goal was reached
            bool isGoalReached();

        private:

            // Callback function for the subscriber to laser scan
            void buildStateRepresentation(sensor_msgs::LaserScan laser_scan);

            bool isCrashed(double& reward);

            bool isAtGoal(double& reward);

            void initializeCustomizedCostmap();

            void initializeTransitionMsg();

            void setZeroAction();

            void addMarkerToArray(double x, double y, std::string frame, ros::Time stamp); // to_delete

            void callbackAction(geometry_msgs::Twist action);

            void addLaserScanPoints(const sensor_msgs::LaserScan& laser_scan);

            void addGlobalPlan();

            void storeResult(const neuro_local_planner_wrapper::Transition& transition);

            void processSubGoal(double& reward);

            bool isGoalInvisible();

            void processGoalVisibility();

            int goal_invisible_count;

            // Listener to get our pose on the map
            tf::TransformListener* tf_;

            // --- Publisher & Subscriber ---

            // For visualisation, publishers of global and local plan
            ros::Publisher g_plan_pub_, l_plan_pub_; // TODO: never used right?

            // Publisher to the stage_sim_bot after crash or reached goal
            ros::Publisher state_pub_;

            // Publisher for velocity commands to neuro_stage_ros for direct controlling
            ros::Publisher action_pub_;

            ros::Subscriber action_sub_;

            // Subscribe to laser scan topic
            ros::Subscriber laser_scan_sub_;

            // For visualisation, publisher for customized costmap
            ros::Publisher customized_costmap_pub_;

            // Publisher for communication with TensorFlow
            ros::Publisher transition_msg_pub_;

            // Publisher for toggling noise for exploration
            ros::Publisher noise_flag_pub_;

            // TODO: remove
            // ros::Publisher debug_marker_pub_;

            // Our costmap ros interface
            costmap_2d::Costmap2DROS* costmap_ros_;

            // Our actual costmap
            costmap_2d::Costmap2D* costmap_;

            // Customized costmap as state representation of the robot base
            nav_msgs::OccupancyGrid customized_costmap_;

            // Indicates whether episode is running or not e.g. reached the goal or crashed
            bool is_running_;

            // Transition message with actual state representation which is four consecutive costmaps stacked together
            // in one vector and actual reward
            neuro_local_planner_wrapper::Transition transition_msg_;

            visualization_msgs::MarkerArray marker_array_; // to_delete

            // last action received from network
            geometry_msgs::Twist action_;

            // Our current pose
            tf::Stamped<tf::Pose> current_pose_;

            // The current global plan in normal and costmap coordinates
            std::vector<geometry_msgs::PoseStamped> global_plan_;

            // Should we use an existing planner plugin to gather samples?
            // Then we need all of these variables...
            bool existing_plugin_;
            pluginlib::ClassLoader<nav_core::BaseLocalPlanner> blp_loader_;
            boost::shared_ptr<nav_core::BaseLocalPlanner> tc_;

            // Are we initialized?
            bool initialized_;

            // Goal counter and crash counter to display in the output
            int goal_counter_, crash_counter_;

            // Array for results
            // time stamp
            std::vector<int> time_stamp_storage_;
            // reward (-1 for collision, +1 for goal reached)
            std::vector<int> reward_storage;

            // To close up an episode if it lasts too long
            double max_time_;
            double start_time_;

            // For plotting
            int temp_time_;
            bool noise_flag_;
            int temp_crash_count_;
            int temp_goal_count_;
            std::vector<std::pair<int, int> > plot_list_;

            int file_counter; // one file for 1000 entries

            long long clock_counter;
    };
};
#endif
