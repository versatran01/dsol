# 🛢️ DSOL: Direct Sparse Odometry Lite

## Reference

Chao Qu, Shreyas S. Shivakumar, Ian D. Miller, Camillo J. Taylor

https://arxiv.org/abs/2203.08182

https://youtu.be/yunBYUACUdg

## Datasets

VKITTI2 https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

KITTI Odom 

TartanAir https://theairlab.org/tartanair-dataset/

Sample realsense data at

https://www.dropbox.com/s/bidng4gteeh8sx3/20220307_172336.bag?dl=0

https://www.dropbox.com/s/e8aefoji684dp3r/20220307_171655.bag?dl=0

## Build

This is a ros package, just put in a catkin workspace and build the workspace.

## Run
Open rviz using the config in `launch/dsol.rviz`

```
roslaunch dsol dsol_data.launch
```

See launch files for more details on different datasets.

See config folder for details on configs.

To run multithread and show timing every 5 frames do
```
roslaunch dsol dsol_data.launch tbb:=1 log:=5
```

## Dependencies

See CMakeLists.txt for dependencies.

## Disclaimer

For reproducing the results in the paper, place use the `iros22` branch.

This is the open-source version, advanced features are not included.
