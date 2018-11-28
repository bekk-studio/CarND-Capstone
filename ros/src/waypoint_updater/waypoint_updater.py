#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import numpy as np

import copy
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

	rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=5)

        # TODO: Add other member variables you need below
        self.pose = None
	self.current_vel = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = -1
	x = [-150., -60., -8., 0.]
	y = [40., 10., 2., 0.]
        self.spl = CubicSpline(x, y, extrapolate=False)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
		#rospy.logwarn('clst_idx: %s, stopline_idx: %s',closest_waypoint_idx, self.stopline_wp_idx) 
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x, y])[1]

        #Check if closest is ahead or behind vehicle/traffic_lights
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[(closest_idx - 1) % len(self.waypoints_2d)]
        
        #We use dot product and scalar projection property
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx
        

    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        farthest_idx = closest_idx + LOOKAHEAD_WPS

	max_index = len(self.base_waypoints.waypoints)
	
	if farthest_idx >= len(self.base_waypoints.waypoints):
	    lane_waypoints = copy.deepcopy(self.base_waypoints.waypoints[closest_idx:])
	else:
            lane_waypoints = copy.deepcopy(self.base_waypoints.waypoints[closest_idx:farthest_idx])
        
        if len(lane_waypoints) < LOOKAHEAD_WPS:
            lane_waypoints.extend(copy.deepcopy(self.base_waypoints.waypoints[0:LOOKAHEAD_WPS-len(lane_waypoints)]))
	

        if self.stopline_wp_idx == -1 or self.stopline_wp_idx - farthest_idx >= 0:
            lane.waypoints = lane_waypoints
        else:
	    lane.waypoints = self.decelerate_waypoints(lane_waypoints, closest_idx)
	    #rospy.logwarn('decelerate')
        
        self.final_waypoints_pub.publish(lane)

    def decelerate_waypoints(self, waypoints, closest_idx):
	stop_idx = max(self.stopline_wp_idx - closest_idx, 0)
	dist = self.distance(waypoints, 0, stop_idx)
        ref_vel = self.get_waypoint_velocity(self.base_waypoints.waypoints[closest_idx])	
        
        if dist < 20. and self.current_vel > 0.9 * ref_vel:
            return waypoints

        if dist > 250.:
	    return waypoints
        
        for i,wp in enumerate(waypoints):
            dist = self.distance(waypoints, i, stop_idx)
            vel = min(float(self.spl(-1*dist)), ref_vel)
	    if vel < 1.:
		vel = 0.
            self.set_waypoint_velocity(waypoints, i, vel)
	  
        return waypoints

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_wp_idx = msg.data

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        if wp1 >= wp2:
	    return 0.
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
