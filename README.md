# DSOL: Direct Sparse Odometry Lite

Sample realsense data at
https://www.dropbox.com/s/bidng4gteeh8sx3/20220307_172336.bag?dl=0
https://www.dropbox.com/s/e8aefoji684dp3r/20220307_171655.bag?dl=0

Clone 
https://github.com/versatran01/ouster_decoder


Open rviz using the config in `launch/dsol.rviz`

Run
```
roslaunch dsol dsol_data.launch
```

See CMakeLists.txt for dependencies.

To run multithread and show timing every 5 frames do
```
roslaunch dsol dsol_data.launch tbb:=1 log:=5
```

This is the open-source version, some advanced features may be missing.

## Reference

Chao Qu, Shreyas S. Shivakumar, Iand D. Miller, Camillo J. Taylor


https://youtu.be/yunBYUACUdg