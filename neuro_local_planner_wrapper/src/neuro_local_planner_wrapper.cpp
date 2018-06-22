#include <neuro_local_planner_wrapper/neuro_local_planner_wrapper.h>
#include <pluginlib/class_list_macros.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(neuro_local_planner_wrapper::NeuroLocalPlannerWrapper, nav_core::BaseLocalPlanner)

double direction_line_len = 15;

int threshold_crash = 170;
double subgoal_reward = 0.2;

namespace neuro_local_planner_wrapper
{

    void DrawArrow(cv::Mat img, cv::Point center, double yaw, cv::Scalar color)
    {
        cv::Point direction = center + cv::Point(cos(yaw)*direction_line_len, sin(yaw)*direction_line_len);
        cv::Point left = center + cv::Point(cos(yaw + M_PI/2.0)*direction_line_len/4, sin(yaw + M_PI/2.9)*direction_line_len/4);
        cv::Point right = center + cv::Point(cos(yaw - M_PI/2.0)*direction_line_len/4, sin(yaw - M_PI/2.9)*direction_line_len/4);
        cv::Point points[3];
        points[0] = direction;
        points[1] = left;
        points[2] = right;
        const cv::Point* ppts[1] = {points};
        int npts[] = {3};

        //cv::rectangle(rawData, center, direction, cv::Scalar(75));
        //cv::ellipse(rawData, center, cv::Size(direction_line_len, direction_line_len/6), yaw*180.0/M_PI, -90, 90, cv::Scalar(75), -1);
        //cv::arrowedLine(rawData, center, direction, cv::Scalar(75), 2, 8, 0, 1.0);
        cv::fillPoly(img, ppts, npts, 1, color);
    }

    void mapToWorld(costmap_2d::Costmap2D* costmap, unsigned int mx, unsigned int my, double& wx, double& wy) {
        wx = costmap->getOriginX() + (mx + 0.5) * costmap->getResolution();
        wy = costmap->getOriginY() + (my + 0.5) * costmap->getResolution();
    }

    bool worldToMap(costmap_2d::Costmap2D* costmap, double wx, double wy, unsigned int& mx, unsigned int& my) {
        double origin_x = costmap->getOriginX(), origin_y = costmap->getOriginY();
        double resolution = costmap->getResolution();

        if (wx < origin_x || wy < origin_y)
            return false;

        mx = (int)((wx - origin_x) / resolution - 0.5);
        my = (int)((wy - origin_y) / resolution - 0.5);

        if (mx < costmap->getSizeInCellsX() && my < costmap->getSizeInCellsY())
            return true;

        return false;
    }

    // Constructor
    NeuroLocalPlannerWrapper::NeuroLocalPlannerWrapper() : initialized_(false), yaw_constraint_flag_(false),
                                                           frame_interval_(4), transition_frame_counter_(0),
                                                           transition_frame_interval_(1), transition_depth_(4),
                                                           blp_loader_("nav_core", "nav_core::BaseLocalPlanner") {}

    // Destructor
    NeuroLocalPlannerWrapper::~NeuroLocalPlannerWrapper()
    {
        tc_.reset();
    }

    // Initialize the planner
    void NeuroLocalPlannerWrapper::initialize(std::string name, tf::TransformListener* tf,
                                         costmap_2d::Costmap2DROS* costmap_ros)
    {
        ROS_WARN(">>>NeuroLocalPlannerWrapper::initialize()");
        // If we are not initialized do so
        if (!initialized_)
        {
            ros::NodeHandle private_nh("~/" + name);

            // TODO: remove
            // debug_marker_pub_ = private_nh.advertise<visualization_msgs::Marker>( "goal_point", 0 );

            // Parameters
            std::string robot_type_str;
            private_nh.param("/robot_type", robot_type_str, std::string("holonomic"));
            yaw_constraint_flag_ = (robot_type_str.compare(std::string("nonholonomic")) == 0);

            private_nh.param("xy_goal_tolerance", xy_goal_tolerance_, 0.1);
            private_nh.param("yaw_goal_tolerance", yaw_goal_tolerance_, 0.1);

            private_nh.param("frame_interval", frame_interval_, 4);
            private_nh.param("transition_frame_interval", transition_frame_interval_, 1);

            ROS_INFO("xy_goal_tolerance: %f", xy_goal_tolerance_);
            ROS_INFO("yaw_goal_tolerance: %f", yaw_goal_tolerance_);
            ROS_INFO("frame_interval_: %d", frame_interval_);
            ROS_INFO("transition_frame_interval_: %d", transition_frame_interval_);


            // Publishers & Subscribers
            state_pub_ = private_nh.advertise<std_msgs::Bool>("new_round", 1);

            //laser_scan_sub_ = private_nh.subscribe("/scan", 1000, &NeuroLocalPlannerWrapper::buildStateRepresentation, this);
            local_costmap_sub_ = private_nh.subscribe("/move_base/local_costmap/costmap", 1000, &NeuroLocalPlannerWrapper::cbLocalCostmap, this);
            local_costmap_update_sub_ = private_nh.subscribe("/move_base/local_costmap/costmap_updates", 1000, &NeuroLocalPlannerWrapper::cbLocalCostmapUpdate, this);

            customized_costmap_pub_ = private_nh.advertise<nav_msgs::OccupancyGrid>("customized_costmap", 1);

            transition_msg_pub_ = private_nh.advertise<neuro_local_planner_wrapper::Transition>("transition", 1);

            action_pub_ = private_nh.advertise<geometry_msgs::Twist>("action", 1);

            action_sub_ = private_nh.subscribe("/neuro_deep_planner/action", 1000, &NeuroLocalPlannerWrapper::callbackAction, this);

            // Setup tf
            tf_ = tf;

            // Setup the costmap_ros interface
            costmap_ros_ = costmap_ros;
            costmap_ros_->getRobotPose(current_pose_);

            // Get the actual costmap object
            costmap_ = costmap_ros_->getCostmap();

            // Initialize customized costmap and transition message
            initializeCustomizedCostmap();
            initializeTransitionMsg();

            // initialize action to zero until first velocity command is computed
            action_ = geometry_msgs::Twist();
            setZeroAction();

            // Should we use the dwa planner?
            existing_plugin_ = false;
            std::string local_planner = "dwa_local_planner/DWAPlannerROS";

            // If we want to, lets load a local planner plugin to do the work for us
            if (existing_plugin_)
            {
                try
                {
                    tc_ = blp_loader_.createInstance(local_planner);
                    ROS_INFO("Created local_planner %s", local_planner.c_str());
                    tc_->initialize(blp_loader_.getName(local_planner), tf, costmap_ros);
                }
                catch (const pluginlib::PluginlibException& ex)
                {
                    ROS_FATAL("Failed to create plugin");
                    exit(1);
                }
            }

            is_running_ = false;

            goal_counter_ = 0;
            crash_counter_ = 0;

            file_counter = 0;

            // For plotting
            noise_flag_ = true;
            temp_time_ = (int)ros::Time::now().toSec();
            temp_crash_count_ = 0;
            temp_goal_count_ = 0;

            // To close up too long episodes
            start_time_ = ros::Time::now().toSec();
            max_time_ = 60*1.5;

            // counter after Goal gets invisible
            goal_invisible_count = 0;
            // We are now initialized
            initialized_ = true;
            clock_counter = 0;
            transition_frame_counter_ = 0;

            if (cost_translation_table_ == NULL)
            {
                cost_translation_table_ = new char[256];

                // special values:
                cost_translation_table_[0] = 0;  // NO obstacle
                cost_translation_table_[253] = 99;  // INSCRIBED obstacle
                cost_translation_table_[254] = 100;  // LETHAL obstacle
                cost_translation_table_[255] = -1;  // UNKNOWN

                // regular cost values scale the range 1 to 252 (inclusive) to fit
                // into 1 to 98 (inclusive).
                for (int i = 1; i < 253; i++)
                {
                    cost_translation_table_[ i ] = char(1 + (97 * (i - 1)) / 251);
                }
            }

            ROS_ERROR("Initialization has been done.");
        }
        else
        {
            ROS_WARN("This planner has already been initialized, doing nothing.");
        }
        ROS_WARN("<<<NeuroLocalPlannerWrapper::initialize()");
    }


    // Sets the plan
    bool NeuroLocalPlannerWrapper::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
    {
        ROS_WARN("<<<NeuroLocalPlannerWrapper::setPlan()");
        // Check if the planner has been initialized
        if (!initialized_)
        {
            ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
            return false;
        }

        // Safe the global plan
        global_plan_.clear();
        global_plan_ = orig_global_plan;

        // If we use the dwa:
        // This code is copied from the dwa_planner
        if (existing_plugin_)
        {
            if (!tc_->setPlan(orig_global_plan))
            {
                ROS_ERROR("Failed to set plan for existing plugin");
                return false;
            }
        }

        // Build global_plan points pose orientation
        for (int i = 0; i < global_plan_.size(); i++)
        {
            int start = std::max(0, i - 6);
            int end = std::min((int)(global_plan_.size() - 2), i + 6);
            std::vector<geometry_msgs::PoseStamped> subpath(&global_plan_[start], &global_plan_[end]);
            double yaw = calculateRotationMomentum(subpath);
            global_plan_[i].pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, yaw); // http://docs.ros.org/api/tf/html/c++/transform__datatypes_8h.html
        }

        goal_position_ = global_plan_.back();

        is_running_ = true;

        return true;
    }


    // Compute the velocity commands
    bool NeuroLocalPlannerWrapper::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        //ROS_WARN("<<<%s()", __FUNCTION__);
        return true;
    }


    // Tell if goal was reached
    bool NeuroLocalPlannerWrapper::isGoalReached()
    {
        double reward = 0.0;
        //ROS_INFO("<<<%s()", __FUNCTION__);
        if (! initialized_) {
            ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
            return false;
        }
        if ( ! costmap_ros_->getRobotPose(current_pose_)) {
            ROS_ERROR("%s: Could not get robot pose", __FUNCTION__);
            return false;
        }

        //if(latchedStopRotateController_.isGoalReached(&planner_util_, odom_helper_, current_pose_)) {
        if(isAtGoal(reward)) {
            ROS_INFO("Goal reached");
            return true;
        } else {
            //ROS_INFO("Goal Not reached");
            return false;
        }
    }


    // Helper function to initialize the state representation
    void NeuroLocalPlannerWrapper::initializeCustomizedCostmap()
    {
        customized_costmap_ = nav_msgs::OccupancyGrid();

        // header
        //customized_costmap_.header.frame_id = "/base_footprint";
        customized_costmap_.header.frame_id = "odom";
        customized_costmap_.header.stamp = ros::Time::now();
        customized_costmap_.header.seq = 0;

        // info
        customized_costmap_.info.width = costmap_->getSizeInCellsX(); // e.g. 80
        customized_costmap_.info.height = costmap_->getSizeInCellsY(); // e.g. 80
        customized_costmap_.info.resolution = (float)costmap_->getResolution(); // e.g. 0.05
        customized_costmap_.info.origin.position.x = -costmap_->getSizeInMetersX()/2.0; // e.g.-1.95
        customized_costmap_.info.origin.position.y = -costmap_->getSizeInMetersY()/2.0; // e.g.-1.95
        customized_costmap_.info.origin.position.z = 0.01; // looks better in simulation
        customized_costmap_.info.origin.orientation.x = 0.0;
        customized_costmap_.info.origin.orientation.y = 0.0;
        customized_costmap_.info.origin.orientation.z = 0.0;
        customized_costmap_.info.origin.orientation.w = 1.0;
    }


    // Helper function to initialize the transition message for the planning node
    void NeuroLocalPlannerWrapper::initializeTransitionMsg()
    {
        for (int i = 0; i < transition_frame_interval_; i++)
        {
            neuro_local_planner_wrapper::Transition transition_msg_;
            // header
            transition_msg_.header.frame_id = customized_costmap_.header.frame_id;
            transition_msg_.header.stamp = customized_costmap_.header.stamp;
            transition_msg_.header.seq = 0;

            // info
            transition_msg_.width = customized_costmap_.info.width;
            transition_msg_.height = customized_costmap_.info.height;
            transition_msg_.depth = transition_depth_; // use four consecutive maps for state representation

            transition_msg_vec_.push_back(transition_msg_);
        }
    }


    // Is called during construction and before the robot is beamed to a new place
    void NeuroLocalPlannerWrapper::setZeroAction()
    {
        action_.linear.x = 0.0;
        action_.linear.y = 0.0;
        action_.linear.z = 0.0;
        action_.angular.x = 0.0;
        action_.angular.y = 0.0;
        action_.angular.z = 0.0;

        action_pub_.publish(action_);
    }


    // Checks if the robot is in collision or not
    bool NeuroLocalPlannerWrapper::isCrashed(double& reward)
    {
        // Get current position of robot
        tf::Stamped<tf::Pose> current_pose;

        if (!costmap_ros_->getRobotPose(current_pose)) // in frame odom
        {
            ROS_ERROR("%s: Could not get robot pose", __FUNCTION__);
            abort();
        }

        // Compute map coordinates
        int robot_x;
        int robot_y;
        costmap_->worldToMapNoBounds(current_pose.getOrigin().getX(), current_pose.getOrigin().getY(), robot_x, robot_y);

        ROS_ERROR("-----(%f, %f) -> (%d, %d)   // (%f, %f)", current_pose.getOrigin().getX(),current_pose.getOrigin().getY(), robot_x, robot_y, costmap_->getOriginX(), costmap_->getOriginY());
        // This causes a crash not just a critical positions but a little bit before the wall
        // TODO: could be solved nicer by using a different inscribed radius, then: >= 253
        if(costmap_->getCost((unsigned int)robot_x, (unsigned int)robot_y) >= threshold_crash)
        {
            crash_counter_++;
            ROS_ERROR("We crashed: %d", crash_counter_);
            reward = -1.0;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool NeuroLocalPlannerWrapper::isTimeOut(double& reward)
    {
        if(ros::Time::now().toSec() - start_time_ > max_time_)
        {
            ROS_ERROR("Time Out");
            return true;
        }
        else
        {
            return false;
        }
    }


    // Checks if the robot reached the goal
    bool NeuroLocalPlannerWrapper::isAtGoal(double& reward)
    {
        // Get current position of robot in odom frame
        costmap_ros_->getRobotPose(current_pose_);

        // Get goal position
        geometry_msgs::PoseStamped goal_position = goal_position_;

        // Transform current position of robot to map frame
        geometry_msgs::PoseStamped current_pose;
        geometry_msgs::PoseStamped current_pose_to_goal_position;
        try
        {
            tf::poseStampedTFToMsg(current_pose_, current_pose);
            tf_->waitForTransform(  goal_position.header.frame_id, current_pose_.frame_id_,
                                    ros::Time(0), ros::Duration(2.0));
            tf_->transformPose(goal_position.header.frame_id, current_pose, current_pose_to_goal_position);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("[failed to transform current pose to map frame to check goal position : %s", ex.what());
        }

        double dist = sqrt(    pow((current_pose_to_goal_position.pose.position.x - goal_position.pose.position.x), 2.0)
                            +   pow((current_pose_to_goal_position.pose.position.y - goal_position.pose.position.y), 2.0));

        bool condition  = dist < xy_goal_tolerance_;
        if (yaw_constraint_flag_ && condition)
        {
            ROS_ERROR("pass xy_goal_tolerance");
            double current_pose_yaw = tf::getYaw(current_pose_to_goal_position.pose.orientation);
            double goal_yaw = tf::getYaw(goal_position.pose.orientation);
            condition &= isSameDirection("goal", current_pose_yaw, goal_yaw);
        }
        // Check if the robot has reached the goal
        if(condition)
        {
            goal_counter_++;
            ROS_ERROR("We reached the goal: %d", goal_counter_);
            reward = 10.0;
            return true;
        }
        else
        {
            return false;
        }
    }

    void NeuroLocalPlannerWrapper::processSubGoal(double& reward)
    {
        if (global_plan_.size() < 2)
        {
            return;
        }

        // Get current position of robot in odom frame
        costmap_ros_->getRobotPose(current_pose_);

        // Get goal position
        geometry_msgs::PoseStamped goal_position = global_plan_.back();

        // Transform current position of robot to map frame
        geometry_msgs::PoseStamped current_pose;
        geometry_msgs::PoseStamped current_pose_to_goal_position;
        try
        {
            tf::poseStampedTFToMsg(current_pose_, current_pose);
            tf_->waitForTransform(  goal_position.header.frame_id, current_pose_.frame_id_,
                                    ros::Time(0), ros::Duration(2.0));
            tf_->transformPose(goal_position.header.frame_id, current_pose, current_pose_to_goal_position);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("[failed to transform current pose to map frame to check subgoal position : %s", ex.what());
        }

        std::vector<geometry_msgs::PoseStamped> global_plan_temp = global_plan_;

        int touched_subgoal_count = 0;
        int touched_max_index = - 1;
        double current_pose_yaw = tf::getYaw(current_pose_to_goal_position.pose.orientation);
        for (int i = global_plan_temp.size() - 2; i >= 0; i--) // skip final goal point
        {
            geometry_msgs::PoseStamped subgoal_position = global_plan_temp[i];
            // Get distance from robot to path point
            double dist = sqrt(     pow((   current_pose_to_goal_position.pose.position.x - subgoal_position.pose.position.x), 2.0)
                               +    pow((   current_pose_to_goal_position.pose.position.y - subgoal_position.pose.position.y), 2.0));

            bool condition  = dist < xy_goal_tolerance_;
            if (yaw_constraint_flag_ && condition)
            {
                double current_pose_yaw = tf::getYaw(current_pose_to_goal_position.pose.orientation);
                double goal_yaw = tf::getYaw(subgoal_position.pose.orientation);
                condition &= isSameDirection("subgoal", current_pose_yaw, goal_yaw);
            }
            // Check if the robot has reached the goal
            if(condition)
            {
                //ROS_INFO("dist: %f", dist);
                //ROS_INFO("We got the sub reward at %d", i);
                reward += subgoal_reward;
                touched_subgoal_count++;
                if (touched_max_index == -1)
                    touched_max_index = i;
            }
        }

        // clear touched path points => construct new plan from untouched path points only
        if (touched_max_index > -1)
        {
            global_plan_.clear();
            for (int i = touched_max_index + 1; i < global_plan_temp.size(); i++)
            {
                global_plan_.push_back(global_plan_temp[i]);
            }
            //ROS_INFO("global_plan_.size(): %d",(int)global_plan_.size());
            //ROS_INFO("We got the sub reward at %f", reward);
            //ROS_INFO("clear touched path points");
        }
    }


    // Publishes the action which is executed by the robot
    void NeuroLocalPlannerWrapper::callbackAction(geometry_msgs::Twist action)
    {
        // Should we use the network as a planner or the dwa planner?
        if (!existing_plugin_)
        {
            // Get action from net
            action_ = action;
        }
        else
        {
            // Use the existing local planner plugin
            geometry_msgs::Twist cmd;
            if(tc_->computeVelocityCommands(cmd))
            {
                if (is_running_) {
                    action_ = cmd;
                }
            }
            else
            {
                ROS_ERROR("Plugin failed computing a command");
            }
        }

        // Publish
        action_pub_.publish(action_);
    }

    void NeuroLocalPlannerWrapper::cbLocalCostmap(nav_msgs::OccupancyGrid grid)
    {
        ROS_WARN(">>>%s()", __FUNCTION__);
        //customized_costmap_.info = grid.info;
        buildStateRepresentation(grid.header, grid.data);
    }

    void NeuroLocalPlannerWrapper::cbLocalCostmapUpdate(map_msgs::OccupancyGridUpdate grid_update)
    {
        ROS_WARN(">>>%s()", __FUNCTION__);
        buildStateRepresentation(grid_update.header, grid_update.data);
    }

    // Callback function for the subscriber to the laser scan
    void NeuroLocalPlannerWrapper::buildStateRepresentation(std_msgs::Header header, std::vector<int8_t> costmap_data)
    {
        ROS_WARN(">>>%s()", __FUNCTION__);
        clock_counter++;
        if (clock_counter%frame_interval_ != 0)
            return;

        transition_frame_counter_++;

        // Check for collision or goal reached
        if (is_running_)
        {
            double reward = 0.0;
            bool          buffer_clear = false;
            unsigned char is_episode_finished = 0;

            // clear costmap/set all pixel gray
            std::vector<int8_t> data(customized_costmap_.info.width*customized_costmap_.info.height,50);
            customized_costmap_.data = data;

            // to_delete: ------
            customized_costmap_.header.stamp = header.stamp;

            {
                double roll, pitch, yaw;
                costmap_ros_->getRobotPose(current_pose_);
                geometry_msgs::PoseStamped current_pose;
                tf::poseStampedTFToMsg(current_pose_, current_pose);
                geometry_msgs::PoseStamped current_pose_to_map;
                try {
                    tf_->waitForTransform(  customized_costmap_.header.frame_id, current_pose.header.frame_id, ros::Time(0), ros::Duration(2.0));
                    tf_->transformPose(     customized_costmap_.header.frame_id, current_pose, current_pose_to_map);
                } catch (tf::TransformException ex) {
                    ROS_ERROR("%s", ex.what());
                }
                //current_pose_.getBasis().getRPY(roll, pitch, yaw);
                yaw = tf::getYaw(current_pose_to_map.pose.orientation);

                // Draw Center
                cv::Mat rawData(cv::Size(customized_costmap_.info.width, customized_costmap_.info.height), CV_8UC1, (void*)&(customized_costmap_.data[0]));

                cv::Point center(customized_costmap_.info.width/2, customized_costmap_.info.height/2);

                DrawArrow(rawData, center, yaw, cv::Scalar(75));

                if (customized_costmap_.header.frame_id.compare("odom") == 0)
                {
                    double resolution = costmap_->getResolution();
                    double wx, wy;
                    costmap_->mapToWorld(0, 0, wx, wy);
                    customized_costmap_.info.origin.position.x = wx - resolution / 2;
                    customized_costmap_.info.origin.position.y = wy - resolution / 2;
                    customized_costmap_.info.origin.position.z = 0.01;
                    customized_costmap_.info.origin.orientation.w = 1.0;
                }
            }

            if (   isCrashed(reward)
                || isTimeOut(reward)
                || isGoalInvisible(reward))
            {
                // New episode so restart the time count
                start_time_ = ros::Time::now().toSec();

                // This is the last transition published in this episode
                is_running_ = false;

                // Stop moving
                setZeroAction();

                // Publish that a new round can be started with the stage_sim_bot
                std_msgs::Bool new_round;
                new_round.data = 1;
                state_pub_.publish(new_round);

                is_episode_finished = 1;
                buffer_clear = true;
                goal_invisible_count = 0;
                clock_counter = 0;
                transition_frame_counter_ = 0;
            }
            else if ( isAtGoal(reward))
            {
                // New episode so restart the time count
                start_time_ = ros::Time::now().toSec();

                // This is the last transition published in this episode
                is_running_ = false;

                // Stop moving
                setZeroAction();

                // Publish that a new round can be started with the stage_sim_bot
                std_msgs::Bool new_round;
                new_round.data = 1;
                state_pub_.publish(new_round);

                is_episode_finished = 0;
                buffer_clear = true;
                goal_invisible_count = 0;
                clock_counter = 0;
                transition_frame_counter_ = 0;
            }
            else{
                // calculated sub goal
                processSubGoal(reward);

                // reset timer
                if (reward > 0.0)
                    start_time_ = ros::Time::now().toSec();
            }

            // add global plan as white pixel with some gradient to indicate its direction
            addGlobalPlan();

            // inspect goal visibility
            processGoalVisibility();

            // add laser scan points as invalid/black pixel
            //addLaserScanPoints(laser_scan);
            {
                boost::unique_lock<costmap_2d::Costmap2D::mutex_t> lock(*(costmap_->getMutex()));

                tf::StampedTransform transform;
                try
                {
                    tf_->lookupTransform(header.frame_id, customized_costmap_.header.frame_id, ros::Time(0), transform);
                }
                catch (tf::TransformException ex)
                {
                    ROS_ERROR("%s", ex.what());
                    return;
                }


                unsigned int mx, my;
                double wx, wy;
                unsigned int value;
                for (unsigned int i = 0; i < customized_costmap_.info.height; i++)
                {
                    for (unsigned int j = 0; j < customized_costmap_.info.width; j++)
                    {
                        wx = customized_costmap_.info.origin.position.x + (j + 0.5)*customized_costmap_.info.resolution;
                        wy = customized_costmap_.info.origin.position.y + (i + 0.5)*customized_costmap_.info.resolution;
                        tf::Point p(wx, wy, 0);
                        p = transform(p);
                        if (costmap_->worldToMap(p.x(), p.y(), mx, my))
                        {
                            value = costmap_->getCost(mx, my);
                            if (value == 255)
                                customized_costmap_.data[i*customized_costmap_.info.width + j] = cost_translation_table_[value];
                            else if (value >= 254)
                                customized_costmap_.data[i*customized_costmap_.info.width + j] = cost_translation_table_[value];
                        }
                    }
                }
/*
                unsigned char* data = costmap_->getCharMap();
                for (unsigned int i = 0; i < customized_costmap_.data.size(); i++)
                {
                    if (data[i] == 255)
                        ;
                    else if (data[i] >= 252)
                        customized_costmap_.data[i] = cost_translation_table_[ data[ i ]];
                }
*/
//                memcpy((void *)customized_costmap_.data.data(), (void *)costmap_->getCharMap(), customized_costmap_.data.size());
//                customized_costmap_.data = costmap_data;
//                for (int i = 0; i < customized_costmap_.data.size(); i++)
//                {
//                    //ROS_ERROR("%d",(unsigned char) customized_costmap_.data[i]);
//                    customized_costmap_.data[i] = local_costmap.data[i];
//                }

            }

            // publish customized costmap for visualization
            customized_costmap_pub_.publish(customized_costmap_);

            // increment seq for next costmap
            customized_costmap_.header.seq = customized_costmap_.header.seq + 1;

            // build transition message/add actual costmap to buffer
            int transition_idx = transition_frame_counter_%transition_frame_interval_;
            neuro_local_planner_wrapper::Transition &transition_msg_ = transition_msg_vec_[transition_idx];

            transition_msg_.state_representation.insert(transition_msg_.state_representation.end(),
                                                        customized_costmap_.data.begin(),
                                                        customized_costmap_.data.end());

            // publish transition message after four consecutive costmaps are available
            if (transition_msg_.state_representation.size() > transition_msg_.width*
                                                            transition_msg_.height*
                                                            transition_msg_.depth)
            {
                // erase first costmap in the queue
                transition_msg_.state_representation.erase(transition_msg_.state_representation.begin(),
                                transition_msg_.state_representation.begin() + transition_msg_.width*transition_msg_.height);

                // publish
                transition_msg_.header.stamp = customized_costmap_.header.stamp;
                transition_msg_.header.frame_id = customized_costmap_.header.frame_id;
                transition_msg_.is_episode_finished = is_episode_finished;
                transition_msg_.reward = reward;

                ROS_WARN("   %s() - publish trainsition", __FUNCTION__);

                transition_msg_pub_.publish(transition_msg_);

                // increment seq for next costmap
                transition_msg_.header.seq = transition_msg_.header.seq + 1;

            }

            // clear buffer
            if (buffer_clear)
            {
                for (int i = 0; i < transition_frame_interval_; i++)
                {
                    transition_msg_vec_[i].state_representation.clear();
                }
            }
        }
        ROS_WARN("<<<%s()", __FUNCTION__);
    }


    // Helper function to generate the transition msg
    void NeuroLocalPlannerWrapper::addLaserScanPoints(const sensor_msgs::LaserScan& laser_scan)
    {
        // get source frame and target frame of laser scan points
        std::string laser_scan_source_frame = laser_scan.header.frame_id;
        std::string laser_scan_target_frame = customized_costmap_.header.frame_id;

        // stamp of first laser point in range
        ros::Time laser_scan_stamp = laser_scan.header.stamp;
        ros::Time customized_costmap_stamp = laser_scan_stamp;

        // update stamp of costmap
        customized_costmap_.header.stamp = customized_costmap_stamp;

        // get transformation between robot base frame and frame of laser scan
        tf::StampedTransform stamped_transform;
        try
        {
            // ros::Time(0) gives us the latest available transform
            tf_->lookupTransform(laser_scan_target_frame, laser_scan_source_frame, ros::Time(0), stamped_transform);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s",ex.what());
        }

        // x and y position of laser scan point in frame of laser scan
        double x_position_laser_scan_frame;
        double y_position_laser_scan_frame;

//        // x and y position of laser scan point in robot base frame
//        double x_position_robot_base_frame;
//        double y_position_robot_base_frame;

        // iteration over all laser scan points
        for(unsigned int i = 0; i < laser_scan.ranges.size(); i++)
        {
            if ((laser_scan.ranges.at(i) > laser_scan.range_min) && (laser_scan.ranges.at(i) < laser_scan.range_max))
            {
                // get x and y coordinates of laser scan point in frame of laser scan, z coordinate is ignored as we
                // are working with a 2D costmap
                x_position_laser_scan_frame = laser_scan.ranges.at(i) * cos(laser_scan.angle_min
                                                                            + i * laser_scan.angle_increment);
                y_position_laser_scan_frame = laser_scan.ranges.at(i) * sin(laser_scan.angle_min
                                                                            + i * laser_scan.angle_increment);

                tf::Point p(x_position_laser_scan_frame, y_position_laser_scan_frame, 0.0);
                p = stamped_transform(p);

                // transformation to costmap coordinates
                int x, y;
                x = (int)round(((p.x() - customized_costmap_.info.origin.position.x)
                                / costmap_->getSizeInMetersX())*customized_costmap_.info.width-0.5);
                y = (int)round(((p.y() - customized_costmap_.info.origin.position.y)
                                / costmap_->getSizeInMetersY())*customized_costmap_.info.height-0.5);


                if ((x >=0) && (y >=0) && (x < customized_costmap_.info.width) && (y < customized_costmap_.info.height))
                {
                    customized_costmap_.data[x + y*customized_costmap_.info.width] = 100;
                }
            }
        }
    }

    void NeuroLocalPlannerWrapper::addGlobalPlan()
    {
        // Transform the global plan into costmap coordinates
        // pose given in fixed frame of global plan which is by default "map"
        geometry_msgs::PoseStamped pose_fixed_frame;

        // pose given in global frame of the local cost map
        geometry_msgs::PoseStamped pose_robot_base_frame;

        std::vector<geometry_msgs::Point> global_plan_map_coordinates;
        geometry_msgs::Point a_global_plan_map_coordinate;

        std::vector<geometry_msgs::PoseStamped> global_plan_temp = global_plan_;

        for(std::vector<geometry_msgs::PoseStamped>::iterator it = global_plan_temp.begin(); it != global_plan_temp.end(); it++) {

            // Transform pose from fixed frame of global plan to global frame of local cost map
            pose_fixed_frame = *it;
            try
            {
                pose_fixed_frame.header.stamp = customized_costmap_.header.stamp;
                tf_->waitForTransform(customized_costmap_.header.frame_id, pose_fixed_frame.header.frame_id,
                                      customized_costmap_.header.stamp, ros::Duration(0.2));
                tf_->transformPose(customized_costmap_.header.frame_id, pose_fixed_frame, pose_robot_base_frame);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
            }

            // transformation to costmap coordinates
            int x, y;
            x = (int)round(((pose_robot_base_frame.pose.position.x - customized_costmap_.info.origin.position.x)
                            / costmap_->getSizeInMetersX()) * customized_costmap_.info.width - 0.5);
            y = (int)round(((pose_robot_base_frame.pose.position.y - customized_costmap_.info.origin.position.y)
                            / costmap_->getSizeInMetersY()) * customized_costmap_.info.height - 0.5);

            if ((x >= 0) && (y >= 0) && (x < customized_costmap_.info.width) && (y < customized_costmap_.info.height))
            {
                a_global_plan_map_coordinate.x = x;
                a_global_plan_map_coordinate.y = y;

                global_plan_map_coordinates.push_back(a_global_plan_map_coordinate);
            }
        }

        // add global plan as white pixels
        for(std::vector<geometry_msgs::Point>::iterator it = global_plan_map_coordinates.begin(); it !=
                global_plan_map_coordinates.end(); it++)
        {
            a_global_plan_map_coordinate = *it;
            customized_costmap_.data[a_global_plan_map_coordinate.x + a_global_plan_map_coordinate.y
                                                                      * customized_costmap_.info.width] = 0;
        }

        // add global plan as bright pixels with gradient
        /*int total_plan_pixel_number = global_plan_map_coordinates.size();
        int counter = 0;
        for(std::vector<geometry_msgs::Point>::iterator it = global_plan_map_coordinates.begin(); it !=
         global_plan_map_coordinates.end(); it++) {
            a_global_plan_map_coordinate = *it;
            customized_costmap_.data[a_global_plan_map_coordinate.x + a_global_plan_map_coordinate.y
            * customized_costmap_.info.width] = 50 - round((double)counter/(double)(total_plan_pixel_number-1)*50.0);
            counter++;
        }*/

        // add global blob
        int goal_tolerance_in_pixel = (int)round(xy_goal_tolerance_ / (costmap_->getSizeInMetersX()
                                                                  / costmap_->getSizeInCellsX()));

#if 0
        geometry_msgs::Point blob_position_map_coordinate;

        bool got_valid_blob_position = false;
        for(std::vector<geometry_msgs::Point>::reverse_iterator it = global_plan_map_coordinates.rbegin(); it !=
                global_plan_map_coordinates.rend(); it++)
        {
            blob_position_map_coordinate = *it;
            if ((blob_position_map_coordinate.x - goal_tolerance_in_pixel >= 0) &&
                (blob_position_map_coordinate.y - goal_tolerance_in_pixel >= 0) &&
                (blob_position_map_coordinate.x + goal_tolerance_in_pixel < customized_costmap_.info.width) &&
                (blob_position_map_coordinate.y + goal_tolerance_in_pixel < customized_costmap_.info.height))
            {
                got_valid_blob_position = true;
                break;
            }
        }

        // goal is is somewhere in the current state representation
        if (got_valid_blob_position)
        {
            int pixel_to_blob_center;
            for (int x = (int)(blob_position_map_coordinate.x - goal_tolerance_in_pixel); x <=
                    blob_position_map_coordinate.x + goal_tolerance_in_pixel; x++)
            {
                for (int y = (int)(blob_position_map_coordinate.y - goal_tolerance_in_pixel); y <=
                        blob_position_map_coordinate.y + goal_tolerance_in_pixel; y++)
                {
                    pixel_to_blob_center = (int)round(sqrt(pow((blob_position_map_coordinate.x - x), 2.0)
                                                           + pow((blob_position_map_coordinate.y  - y), 2.0)));

                    if (pixel_to_blob_center <= goal_tolerance_in_pixel)
                    {
                        customized_costmap_.data[x + y*customized_costmap_.info.width] = 0;
                    }
                }
            }
        }
        else // goal is outside of the current state representation
        {

        }
#else
        geometry_msgs::PoseStamped goal_position = global_plan_.back();
        geometry_msgs::PoseStamped goal_pose_robot_base_frame;
        int goal_x, goal_y;

        try
        {
            goal_position.header.stamp = customized_costmap_.header.stamp;
            tf_->waitForTransform(customized_costmap_.header.frame_id, goal_position.header.frame_id,
                                  customized_costmap_.header.stamp, ros::Duration(0.2));
            tf_->transformPose(customized_costmap_.header.frame_id, goal_position, goal_pose_robot_base_frame);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
        }

        // transformation to costmap coordinates
        goal_x = (int)round(((goal_pose_robot_base_frame.pose.position.x - customized_costmap_.info.origin.position.x)
                        / costmap_->getSizeInMetersX()) * customized_costmap_.info.width - 0.5);
        goal_y = (int)round(((goal_pose_robot_base_frame.pose.position.y - customized_costmap_.info.origin.position.y)
                        / costmap_->getSizeInMetersY()) * customized_costmap_.info.height - 0.5);

        int pixel_to_blob_center;
//        for (int x = (int)(goal_x - goal_tolerance_in_pixel); x <=
//                goal_x + goal_tolerance_in_pixel; x++)
//        {
//            for (int y = (int)(goal_y - goal_tolerance_in_pixel); y <=
//                    goal_y + goal_tolerance_in_pixel; y++)
//            {
//                pixel_to_blob_center = (int)round(sqrt(pow((goal_x - x), 2.0)
//                                                       + pow((goal_y  - y), 2.0)));
//
//                if ((x >= 0) && (y >= 0) && (x < customized_costmap_.info.width) && (y < customized_costmap_.info.height))
//                {
//                    if (pixel_to_blob_center <= goal_tolerance_in_pixel)
//                    {
//                        customized_costmap_.data[x + y*customized_costmap_.info.width] = 0;
//                    }
//                }
//            }
//        }
        geometry_msgs::PoseStamped current_pose;
        tf::poseStampedTFToMsg(current_pose_, current_pose);
        geometry_msgs::PoseStamped goal_pose_to_map;
        try
        {
            tf_->waitForTransform(  customized_costmap_.header.frame_id, goal_position.header.frame_id,
                                    ros::Time(0), ros::Duration(2.0));
            tf_->transformPose(customized_costmap_.header.frame_id, goal_position, goal_pose_to_map);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
        }
        cv::Mat rawData(cv::Size(customized_costmap_.info.width, customized_costmap_.info.height), CV_8UC1, (void*)&(customized_costmap_.data[0]));
        double goal_yaw = tf::getYaw(goal_pose_to_map.pose.orientation);
        cv::Point goal_to_map(goal_x, goal_y);

        DrawArrow(rawData, goal_to_map, goal_yaw, cv::Scalar(0));
#endif
    }

    bool NeuroLocalPlannerWrapper::isGoalInvisible(double& reward)
    {
        if (goal_invisible_count > 8)
        {
            ROS_ERROR("Goal is invisible!");
            return true;
        }
        return false;
    }

    void NeuroLocalPlannerWrapper::processGoalVisibility()
    {
        bool found = false;
        for (int i = 0; i < customized_costmap_.data.size(); i++)
        {
            if (customized_costmap_.data[i] == 0)
            {
                found = true;
                break;
            }
        }

        // if current local map
        if (!found)
        {
            goal_invisible_count++;    // increase count
            ROS_WARN("Goal is invisible... count=%d", goal_invisible_count);
        }
        else
            goal_invisible_count = 0;  // reset
    }

    double NeuroLocalPlannerWrapper::calculateRotationMomentum(std::vector<geometry_msgs::PoseStamped> subpath)
    {
        double yaw = 0.0;
        if (subpath.size() < 2)
        {
            return 0.0;
        }
        geometry_msgs::Vector3 vec;
        for (int i = 0; i < subpath.size() - 1; i++)
        {
            geometry_msgs::PoseStamped cur = subpath[i];
            geometry_msgs::PoseStamped next = subpath[i + 1];

            vec.x += next.pose.position.x - cur.pose.position.x;
            vec.y += next.pose.position.y - cur.pose.position.y;
        }

        yaw = std::atan2(vec.y, vec.x);
        // normalize
        double length = sqrt(vec.x*vec.x + vec.y*vec.y);
        return yaw;
    }

    bool NeuroLocalPlannerWrapper::isSameDirection(std::string label, double yaw1, double yaw2)
    {
        double yaw_diff = fabs(yaw1 - yaw2);
        //ROS_ERROR("### %s yaw diff = %f (%f - %f)", label.c_str(), yaw_diff, yaw1, yaw2);
        return ((yaw_diff < yaw_goal_tolerance_) || (yaw_diff > 2*3.14 - yaw_goal_tolerance_)) ;
    }
};
