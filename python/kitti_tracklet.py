#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# import parseTrackletXML as xmlParser



from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
import numpy as np
from warnings import warn

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0':STATE_UNSET, '1':STATE_INTERP, '2':STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1':OCC_UNSET, '0':OCC_VISIBLE, '1':OCC_PARTLY, '2':OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99':TRUNC_UNSET, '0':TRUNC_IN_IMAGE, '1':TRUNC_TRUNCATED, '2':TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}

class Tracklet(object):
    """ 
    Representation an annotated object track 
    Tracklets are created in function parseXML and can most conveniently used as follows:
    for trackletObj in parseXML(trackletFile):
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
      ... your code here ...
    #end: for all frames
    #end: for all tracklets
    absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
    amtOcclusion and amtBorders could be None
    You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int), 
    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
    """

    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None   # n x 3 float array (x,y,z)
    rots = None    # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        """ 
        Creates Tracklet with no info set 
        """
        self.size = np.nan*np.ones(3, dtype=float)

    def __str__(self):
        """ 
        Returns human-readable string representation of tracklet object
        called implicitly in 
        print trackletObj
        or in 
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

    def __iter__(self):
        """ 
        Returns an iterator that yields tuple of all the available data for each frame 
        called whenever code iterates over a tracklet object, e.g. in 
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amtOccs is None:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None), range(self.firstFrame, self.firstFrame+self.nFrames))
        else:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs,
                self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame+self.nFrames))
#end: class Tracklet


def parseXML(trackletFile):
    """ 
    Parses tracklet xml file and convert results to list of Tracklet objects
    :param trackletFile: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """

    # convert tracklet XML data to a tree structure
    eTree = ElementTree()
    print('Parsing tracklet file', trackletFile)
    with open(trackletFile) as f:
        eTree.parse(f)

    # now convert output to list of Tracklet objects
    trackletsElem = eTree.find('tracklets')
    tracklets = []
    trackletIdx = 0
    nTracklets = None
    for trackletElem in trackletsElem:
        #print 'track:', trackletElem.tag
        if trackletElem.tag == 'count':
            nTracklets = int(trackletElem.text)
            print('File contains', nTracklets, 'tracklets')
        elif trackletElem.tag == 'item_version':
            pass
        elif trackletElem.tag == 'item':
            #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
            # a tracklet
            newTrack = Tracklet()
            isFinished = False
            hasAmt = False
            frameIdx = None
            for info in trackletElem:
                # print('trackInfo:', info.tag)
                if isFinished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    newTrack.objectType = info.text
                elif info.tag == 'h':
                    newTrack.size[0] = float(info.text)
                elif info.tag == 'w':
                    newTrack.size[1] = float(info.text)
                elif info.tag == 'l':
                    newTrack.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    newTrack.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        #print 'trackInfoPose:', pose.tag
                        if pose.tag == 'count':     # this should come before the others
                            if newTrack.nFrames is not None:
                                raise ValueError('there are several pose lists for a single track!')
                            elif frameIdx is not None:
                                raise ValueError('?!')
                            newTrack.nFrames = int(pose.text)
                            newTrack.trans = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.rots = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.occs = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
                            newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
                            newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            frameIdx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frameIdx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                #print 'trackInfoPoseInfo:', poseInfo.tag
                                if poseInfo.tag == 'tx':
                                    newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                                    hasAmt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frameIdx += 1
                        else:
                            raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    isFinished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
            #end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not isFinished:
                warn('tracklet {0} was not finished!'.format(trackletIdx))
            if newTrack.nFrames is None:
                warn('tracklet {0} contains no information!'.format(trackletIdx))
            elif frameIdx != newTrack.nFrames:
                warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(trackletIdx, newTrack.nFrames, frameIdx))
            if np.abs(newTrack.rots[:,:2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amtOccs / amtBorders are not set, set them to None
            if not hasAmt:
                newTrack.amtOccs = None
                newTrack.amtBorders = None

            # add new tracklet to list
            tracklets.append(newTrack)
            trackletIdx += 1

        else:
            raise ValueError('unexpected tracklet info')
    #end: for tracklet list items

    print('Loaded', trackletIdx, 'tracklets.')

    # final consistency check
    if trackletIdx != nTracklets:
        warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets, trackletIdx))

    return tracklets


def publishTracklet(input:PointCloud2):
    global framecount,tracklet
    header = input.header
    markerArray = MarkerArray()
    
    # delete all previousframe object
    marker_deleteall = Marker()
    marker_deleteall.action = Marker.DELETEALL
    markerArray.markers.append(marker_deleteall)


    for trackletObj in tracklet:
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
            if absoluteFrameNumber == framecount:
                marker = Marker()
                marker.header= header
                marker.ns = 'box'
                marker.id = tracklet.index(trackletObj)
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.scale.x = trackletObj.size[0]
                marker.scale.y = trackletObj.size[1]
                marker.scale.z = trackletObj.size[2]
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.position.x = translation[0]
                marker.pose.position.y = translation[1]
                marker.pose.position.z = translation[2]
                marker.lifetime = rospy.Duration()


                marker_text = Marker()
                marker_text.header = header
                marker_text.id = tracklet.index(trackletObj)
                marker_text.ns = 'text'
                marker_text.text = f'ID: {tracklet.index(trackletObj)}'
                marker_text.type = Marker.TEXT_VIEW_FACING
                marker_text.action = Marker.ADD
                marker_text.scale.z = 0.5
                marker_text.color.a = 1.0
                marker_text.color.r = 0.0
                marker_text.color.g = 0.0
                marker_text.color.b = 0.0
                marker_text.pose.orientation.w = 1.0
                marker_text.pose.orientation.x = 0.0
                marker_text.pose.orientation.y = 0.0
                marker_text.pose.orientation.z = 0.0
                marker_text.pose.position.x = translation[0]
                marker_text.pose.position.y = translation[1]
                marker_text.pose.position.z = translation[2]+5
                marker_text.lifetime = rospy.Duration()

                markerArray.markers.append(marker_text)

                markerArray.markers.append(marker)
    
    rospy.loginfo(f'frame: {framecount}')
    framecount = framecount+1
    rospy.loginfo(f'Publishing {len(markerArray.markers)} Objects')
    pub_kitti_tracklet.publish(markerArray)

if __name__ == "__main__":
    framecount = 0 
    Tracklet_Dir = 'tracklet_labels.xml'
    tracklet = parseXML(Tracklet_Dir)
    print(len(tracklet))
    rospy.init_node('listener',anonymous= True)
    rospy.Subscriber("/kitti/velo/pointcloud",PointCloud2,publishTracklet)
    pub_kitti_tracklet = rospy.Publisher('kitti_tracklet',MarkerArray,queue_size=10)
    rospy.spin()
