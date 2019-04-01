# -*- coding: utf-8 -*
import arcpy
import copy
import numpy as np
import sys
import os
class Data:
    def __init__(self):
        self.havedRows=[]
        self.havedFid=[]
        self.isStart=True
class QuickFind(object):
    def __init__(self, n):
        self.element_id = []
        self.count = len(n)
        i = 0
        while i < self.count:
            self.element_id.append([n[i],i])
            i += 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def find(self, p):
        for i in self.element_id:
            if(i[0]==p):
                return i[1]
        return None

    def union(self, p, q):
        idp = self.find(p)
        idq = self.find(q)
        if not self.connected(p, q):
            for i in self.element_id:
                if(i[1]==idp):
                    i[1]=idq
            self.count -= 1
    def getClassifyInfo(self):
        ClassifyInfo=[]
        ids=list(set([k[1] for k in self.element_id]))
        for i in ids:
            classify=[]
            for j in self.element_id:
                if(j[1]==i):
                    classify.append(j[0])
            ClassifyInfo.append(classify)
        return ClassifyInfo
def getMultiAngles(angle_i,pts2):
    multi_angles = [];single_angles = []
    for j in range(len(pts2) - 1):
        if (pts2[j][0] == pts2[j + 1][0]):
            angle_j = 90
        else:
            angle_j = np.arctan((pts2[j][1] - pts2[j + 1][1]) / float(pts2[j][0] - pts2[j + 1][0])) * 180 / np.pi
        if (angle_j < 0):
            angle_j += 180
        if(len(single_angles)==0):
            if (abs(angle_j - angle_i) < 10 or abs(angle_j - angle_i) > 170):
                single_angles.append([j, angle_j,pts2[j][0],pts2[j][1]])
            else:
                continue
        else:
            if (abs(angle_j - single_angles[-1][1]) < 10 or abs(angle_j - single_angles[-1][1]) > 170):
                single_angles.append([j, angle_j,pts2[j][0],pts2[j][1]])
            else:
                multi_angles.append(single_angles)
                single_angles=[]
                if (abs(angle_j - angle_i) < 10 or abs(angle_j - angle_i) > 170):
                    single_angles.append([j, angle_j,pts2[j][0],pts2[j][1]])
                else:
                    continue
        if(j==(len(pts2) - 2) and len(single_angles)!=0):
            multi_angles.append(single_angles)
    return multi_angles

def getAllData(pts1,pts2):
    pts1_nums_angles=[]
    all_data=[]
    flag=0
    for i in range(len(pts1)-1):
        if(pts1[i][0]==pts1[i+1][0]):
            angle_i=90
        else:
            angle_i=np.arctan((pts1[i][1]-pts1[i+1][1])/float(pts1[i][0]-pts1[i+1][0]))*180/np.pi
        if (angle_i < 0):
            angle_i += 180
        if(flag==0):
            multi_angles = getMultiAngles(angle_i, pts2)
            if(len(multi_angles)==0):
                continue
            else:
                flag=1
        if(len(pts1_nums_angles)==0):
            pts1_nums_angles.append([i,angle_i,pts1[i][0],pts1[i][1]])
        else:
            flag2=0
            for single_angle in multi_angles:
                if(abs(angle_i-single_angle[0][1])<10 or abs(angle_i-single_angle[0][1])>170):
                    pts1_nums_angles.append([i, angle_i, pts1[i][0], pts1[i][1]])
                    flag2=1
                    break
            if(flag2==0):
                all_data.append([pts1_nums_angles, multi_angles])
                pts1_nums_angles = []
                multi_angles = getMultiAngles(angle_i, pts2)
                if (len(multi_angles) == 0):
                    pts1_nums_angles = []
                    multi_angles = []
                    flag = 0
                else:
                    pts1_nums_angles.append([i, angle_i, pts1[i][0], pts1[i][1]])
                    flag = 1
        if(i==(len(pts1)-2) and len(multi_angles)!=0):
            all_data.append([pts1_nums_angles, multi_angles])
    return all_data,pts1,pts2
def processIntersect(row1,row2):
    pts1=getPts(row1);pts2=getPts(row2)
    pts1_x=[i[0] for i in pts1];pts1_y=[i[1] for i in pts1]
    pts2_x = [i[0] for i in pts2];pts2_y = [i[1] for i in pts2]
    pts12_x=[];pts12_y=[]
    pts12_x.extend(pts1_x);pts12_x.extend(pts2_x)
    pts12_y.extend(pts1_y);pts12_y.extend(pts2_y)
    condition1=abs(pts1[0][0]-pts2[0][0])<2 and abs(pts1[0][1]-pts2[0][1])<2
    condition2=abs(pts1[-1][0]-pts2[-1][0])<2 and abs(pts1[-1][1]-pts2[-1][1])<2
    condition3=abs(pts1[0][0]-pts2[-1][0])<2 and abs(pts1[0][1]-pts2[-1][1])<2
    condition4 = abs(pts1[-1][0] - pts2[0][0]) < 2 and abs(pts1[-1][1] - pts2[0][1]) < 2
    if((condition1 and condition2) or (condition3 and condition4)):
        distMax=0
        for i in pts1:
            for j in pts2:
                dist=np.linalg.norm(np.array(i)-np.array(j))
                if(distMax<dist):
                    distMax=dist
        if(distMax<65):
            return True,True,distMax
        else:
            return False,False,distMax
    else:
        if (condition1):
            haha1 = pts1[len(pts1) - 1][0] > pts1[0][0]+10 and pts2[len(pts2) - 1][0] > pts1[0][0]+10
            haha2 = pts1[len(pts1) - 1][1] > pts1[0][1]+10 and pts2[len(pts2) - 1][1] > pts1[0][1]+10
            haha3 = pts1[len(pts1) - 1][0]+10 < pts1[0][0] and pts2[len(pts2) - 1][0]+10 < pts1[0][0]
            haha4 = pts1[len(pts1) - 1][1]+10 < pts1[0][1] and pts2[len(pts2) - 1][1]+10 < pts1[0][1]
            a=getAngle([0,row1],[0,row2])
            haha5 = a < 25
            dist = np.sqrt((pts1[len(pts1) - 1][0] - pts2[len(pts2) - 1][0]) ** 2 + (
                        pts1[len(pts1) - 1][1] - pts2[len(pts2) - 1][1]) ** 2)
            if((haha1 or haha2 or haha3 or haha4) and haha5 and dist<65):
                return True,True,dist
            elif((haha1 or haha2 or haha3 or haha4)  and dist<65):
                return False,True,dist
            else:
                return False,False,dist
        elif(condition3):
            haha1 = pts1[len(pts1) - 1][0] > pts1[0][0]+10 and pts2[0][0] > pts1[0][0]+10
            haha2 = pts1[len(pts1) - 1][1] > pts1[0][1]+10 and pts2[0][1] > pts1[0][1]+10
            haha3 = pts1[len(pts1) - 1][0]+10 < pts1[0][0] and pts2[0][0]+10 < pts1[0][0]
            haha4 = pts1[len(pts1) - 1][1]+10 < pts1[0][1] and pts2[0][1]+10 < pts1[0][1]
            a = getAngle([0,row1], [-1,row2])
            haha5 =a < 25
            dist = np.sqrt((pts1[len(pts1) - 1][0] - pts2[0][0])**2+(pts1[len(pts1) - 1][1] - pts2[0][1]) ** 2)
            if((haha1 or haha2 or haha3 or haha4) and haha5 and dist<65):
                return True,True,dist
            elif((haha1 or haha2 or haha3 or haha4)  and dist<65):
                return False,True,dist
            else:
                return False,False,dist
        elif(condition4):
            haha1 = pts1[0][0] > pts2[0][0]+10 and pts2[len(pts2)-1][0] > pts2[0][0]+10
            haha2 = pts1[0][1] > pts2[0][1]+10 and pts2[len(pts2)-1][1] > pts2[0][1]+10
            haha3 = pts1[0][0]+10 < pts2[0][0] and pts2[len(pts2)-1][0]+10 < pts2[0][0]
            haha4 = pts1[0][1]+10 < pts2[0][1] and pts2[len(pts2)-1][1]+10 < pts2[0][1]
            a = getAngle([-1,row1],[0,row2])
            haha5 = a < 25
            dist = np.sqrt((pts1[0][0] - pts2[len(pts2)-1][0]) ** 2 + (pts1[0][1] - pts2[len(pts2)-1][1]) ** 2)
            if((haha1 or haha2 or haha3 or haha4) and haha5 and dist<65):
                return True,True,dist
            elif((haha1 or haha2 or haha3 or haha4)  and dist<65):
                return False,True,dist
            else:
                return False,False,dist
        elif(condition2):
            haha1 = pts1[0][0] > pts2[len(pts2)-1][0]+10 and pts2[0][0] > pts2[len(pts2)-1][0]+10
            haha2 = pts1[0][1] > pts2[len(pts2)-1][1]+10 and pts2[0][1] > pts2[len(pts2)-1][1]+10
            haha3 = pts1[0][0]+10 < pts2[len(pts2)-1][0] and pts2[0][0]+10 < pts2[len(pts2)-1][0]
            haha4 = pts1[0][1]+10 < pts2[len(pts2)-1][1] and pts2[0][1]+10 < pts2[len(pts2)-1][1]
            a = getAngle([-1,row1],[-1,row2])
            haha5 = a < 25
            dist = np.sqrt((pts1[0][0] - pts2[0][0]) ** 2 + (pts1[0][1] - pts2[0][1]) ** 2)
            if ((haha1 or haha2 or haha3 or haha4) and haha5 and dist < 65):
                return True, True,dist
            elif ((haha1 or haha2 or haha3 or haha4) and dist < 65):
                return False,True,dist
            else:
                return False,False,dist
        else:
            return None,None,0

def computeAngle(pts):
    pt1=pts[0];pt2=pts[len(pts)-1]
    if(pt1[0]==pt2[0]):
        angle=90
    else:
        angle=np.arctan((pt1[1]-pt2[1])/float(pt1[0]-pt2[0]))*180/np.pi
        if(angle<0):
            angle+=180
    return angle

def getDistSum(pts):
    distSum=0
    for i in range(len(pts)-1):
        distSum+=np.linalg.norm(np.array(pts[i])-np.array(pts[i+1]))
    return distSum
def getProjectPoint(p0,p1,p2):
    vec_p0_p1=np.array(p0)-np.array(p1)
    vec_p2_p1=np.array(p2)-np.array(p1)
    k=np.dot(vec_p0_p1,vec_p2_p1)/(np.linalg.norm(vec_p2_p1)**2)
    p3=k*vec_p2_p1+np.array(p1)
    return p3
def concentratedLine(line):
    iter=[i*0.1 for i in range(10)]
    addPts=[]
    for i in range(len(line)-1):
        tmp=[]
        for j in iter:
            px=j*(np.array(line[i+1])-np.array(line[i]))+np.array(line[i])
            tmp.append(px.tolist())
        addPts.extend(tmp)
    addPts.append(line[-1])
    return addPts
def isOverLap(line1_tmp,line2_tmp,pts1,pts2,distt):
    line1=concentratedLine(line1_tmp)
    line2=concentratedLine(line2_tmp)
    overlap_line1=[];overlap_line2=[]
    for i in range(len(line1)):
        p=getProjectPoint(line1[i],line2[0],line2[-1])
        dist_p_line2_start=np.linalg.norm(p-np.array(line2[0]))
        dist_p_line2_end = np.linalg.norm(p-np.array(line2[-1]))
        condition=min(dist_p_line2_start,dist_p_line2_end)<20
        dist=np.linalg.norm(p-np.array(line1[i]))
        if(((np.array(line2[0])-p)[0]*(np.array(line2[-1])-p)[0]<0 or condition) and dist<65):
            overlap_line1.append(line1[i])
    for i in range(len(line2)):
        p = getProjectPoint(line2[i], line1[0], line1[-1])
        dist_p_line1_start=np.linalg.norm(p-np.array(line1[0]))
        dist_p_line1_end = np.linalg.norm(p-np.array(line1[-1]))
        condition=min(dist_p_line1_start,dist_p_line1_end)<20
        dist = np.linalg.norm(p - np.array(line2[i]))
        if (((np.array(line1[0]) - p)[0] * (np.array(line1[-1]) - p)[0] < 0 or condition) and dist < 65):
            overlap_line2.append(line2[i])
    overlap_line1_ratio=getDistSum(overlap_line1)/getDistSum(pts1)
    overlap_line2_ratio=getDistSum(overlap_line2)/getDistSum(pts2)
    if((overlap_line1_ratio>0.5 or overlap_line2_ratio>0.5) and distt<65):
        return True
    elif (overlap_line1_ratio > 0.3 and overlap_line2_ratio > 0.3 and distt < 65):
        return True
    elif(overlap_line1_ratio>0.8 or overlap_line2_ratio>0.8):
        return True
    else:
        return False

def getPts(row):
    pts=[]
    for part in row[1]:
        for point in part:
            pts.append([point.X,point.Y])
    return pts

def isTrueData(all_data,pts1,pts2,dist):
    if (len(all_data) != 0):
        for i in all_data:
            pts1_tmp=[];multi_angles = i[1]
            for m in i[0]:
                pts1_tmp.append(pts1[m[0]])
            pts1_tmp.append(pts1[i[0][-1][0]+1])
            for single_angles in multi_angles:
                pts2_tmp=[]
                for n in single_angles:
                    pts2_tmp.append(pts2[n[0]])
                pts2_tmp.append(pts2[single_angles[-1][0] + 1])
                if (isOverLap(pts1_tmp, pts2_tmp,pts1,pts2,dist)):
                    return True
        return False
    else:
        return False
def maxAngle(pts):
    Angles=[]
    for i in range(len(pts)-1):
        pt1 = pts[i];pt2 = pts[i+1]
        if (pt1[0] == pt2[0]):
            angle = 90
        else:
            angle = np.arctan((pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) * 180 / np.pi
            if (angle < 0):
                angle += 180
        Angles.append(angle)
    return max(Angles)
def getMatchRows(rows_process_step1,row):
    rows_process_step2_intersect=[]
    rows_process_step2=[]
    row_pts=getPts(row)
    for i in rows_process_step1:
        # if(i[0]!=488):
        #      continue
        pts=getPts(i)
        distt=np.linalg.norm(np.array(pts[0])-np.array(pts[-1]))
        if(distt<2):
            continue
        isintersect1,isintersect2,dist=processIntersect(row,i)
        if (isintersect1 and isintersect2):
            rows_process_step2_intersect.append(i)
        else:
            all_data, pts1, pts2 = getAllData(row_pts, pts)
            isTrue=isTrueData(all_data, pts1, pts2,dist)
            if( isTrue and not isintersect1 and isintersect2):
                rows_process_step2_intersect.append(i)
            elif(isTrue and (isintersect1==None)):
                rows_process_step2.append(i)
            else:
                pass
    rows_process_step2_intersect_index=[]
    result = [];result_index=[]
    if(len(rows_process_step2_intersect)>0):
            rows_process_step2_intersect_index=[k[0] for k in rows_process_step2_intersect]
            result_index.extend(rows_process_step2_intersect_index)
            result.extend(rows_process_step2_intersect)
    for m in rows_process_step2:
            result.append(m)
            result_index.append(m[0])
    return result, result_index, rows_process_step2_intersect_index
def getTwoWayStreet(infoAllRows,row):
    left_fid=row[2];right_fid=row[3]
    rows_process_step1=[]
    for row_tmp in infoAllRows:
        if(row_tmp[2]==left_fid and row_tmp[3]==right_fid):
            continue
        if(row_tmp[2]==left_fid or row_tmp[2]==right_fid or row_tmp[3]==left_fid or row_tmp[3]==right_fid):
            rows_process_step1.append(row_tmp)
    allMatchedRows=[];allMatchedRowsIntersect=[]
    HavedMatchedIndex=[]
    while(1):
        result, result_index, result_intersect_index = getMatchRows(rows_process_step1, row)
        HavedMatchedIndex.extend(result_index)
        if(len(result)==0):
            break
        allMatchedRows.extend(result)
        if(len(result_intersect_index)!=0):
            allMatchedRowsIntersect.extend(result_intersect_index)
        connectRows = []
        for i in result:
            pts_i = getPts(i)
            for j in infoAllRows:
                if (j[0] == i[0]):
                    continue
                pts_j = getPts(j)
                if (isSim(pts_i, pts_j)[0] != None and j[0] not in HavedMatchedIndex and j[0]!=row[0]):
                    connectRows.append(j)
        rows_process_step1=connectRows
    allMatchedRowsFids=[k[0] for k in allMatchedRows]
    return allMatchedRows,allMatchedRowsFids,allMatchedRowsIntersect

def processRows(result):
    num=0
    remainLines=[]
    for i in result:
        pts_i=getPts(i)
        start=0;end=0
        for j in result:
            if(i[0]==j[0]):
                continue
            pts_j=getPts(j)
            start_index,end_index=isSim(pts_i,pts_j)
            if(start_index!=None):
                if(start_index==0):
                    start+=1
                else:
                    end+=1
        if(start==0 or end==0):
            remainLines.append(i)
        if(start==0):
            num+=1
        if(end==0):
            num+=1
    if(num<4):
        if(len(result)==2):
            pts0=getPts(result[0]);pts1=getPts(result[1])
            index1,_=isSim(pts0,pts1)
            if(index1!=None):
                return True,result
        else:
            return False,[]
    fids=[]
    for i in remainLines:
        fids.extend([i[2],i[3]])
    fid_max_index=max(fids,key=fids.count)
    finally_result=[]
    for j in remainLines:
        if(j[2]==fid_max_index or j[3]==fid_max_index):
            finally_result.append(j)
    return True,finally_result

def getAngle(data_i,data_j):
    pts_i=getPts(data_i[1])
    pts_j=getPts((data_j[1]))
    pts_i_vec=np.array(pts_i[getIndex(data_i[0])])-np.array(pts_i[data_i[0]])
    pts_j_vec=np.array(pts_j[getIndex(data_j[0])])-np.array(pts_j[data_j[0]])
    pts_i_vec_norm=np.linalg.norm(pts_i_vec)
    pts_j_vec_norm=np.linalg.norm(pts_j_vec)
    cosAngle=np.dot(pts_i_vec,pts_j_vec)/(pts_i_vec_norm*pts_j_vec_norm)
    Angle = np.arccos(cosAngle) * 180 / np.pi
    return Angle

def isGoodGroupLines(pts_i,pts_j):
    pts_i=concentratedLine(pts_i)
    dist=np.linalg.norm(np.array(pts_j[0])-np.array(pts_j[-1]))
    num=0
    for i in pts_i[1:-1]:
        a=np.array(i)-np.array(pts_j[0])
        b=np.array(pts_j[-1])-np.array(pts_j[0])
        tmp=abs(np.cross(a,b))
        tmp=tmp/dist
        if(tmp>1):
            num+=1
    if(num==0):
        return True
    else:
        return False

def DisplayPogressBar(index,numsRows):
    sys.stdout.write("\r" + "[%s%%]" % ((index + 1) * 100 / numsRows))
    sys.stdout.flush()
def addField(shpPath):
    infoAllRows=[]
    cursor = arcpy.da.SearchCursor(shpPath, ["OID@", "SHAPE@", "LEFT_FID", "RIGHT_FID"])
    for i in cursor:
        infoAllRows.append(i)
    info_rows_after1=[];info_rows_after2=[]
    info_rows=[]
    info=[]
    TwoWayInfo={}
    index=0
    numsRows=len(infoAllRows)
    for row in infoAllRows:
        pts=getPts(row)
        dist=np.linalg.norm(np.array(pts[0])-np.array(pts[-1]))
        if(dist<2):
            index += 1
            DisplayPogressBar(index,numsRows)
            continue
        result, result_index, result_intersect_index = getTwoWayStreet(infoAllRows,row)
        if(len(result_index)>0):
            info.extend(result_index)
            info.append(row[0])
            info_rows.append(row)
            info_rows.extend(result)
            TwoWayInfo[row]=[result, result_index, result_intersect_index]
        index+=1
        DisplayPogressBar(index, numsRows)
    for i in info_rows:
        if (len(info_rows_after1) == 0):
            info_rows_after1.append(i)
        else:
            info_index = [m[0] for m in info_rows_after1]
            if (not i[0] in info_index):
                info_rows_after1.append(i)

        if (len(info_rows_after2) == 0):
            info_rows_after2.append(i)
        else:
            info_index = [m[0] for m in info_rows_after2]
            if (not i[0] in info_index):
                info_rows_after2.append(i)
    print "\ndone"
    return info_rows_after1,info_rows_after2,TwoWayInfo

def getIntersectPts(pts1,pts2):
    if(pts1[0]==pts2[0] or pts1[0]==pts2[-1]):
        return 0
    if(pts1[-1]==pts2[0] or pts1[-1]==pts2[-1]):
        return -1
    return None
def deleteRows_Test(data,TwoWayInfo):
    intersectPts=[]
    intersectFids=[]
    for i in data.havedRows:
            pts_i=getPts(i)
            try:
                result, _, result_intersect = TwoWayInfo[i]
            except:
                continue
            if(len(result_intersect)!=0):
                result = filter(lambda x: x[0] in result_intersect, result)
                for j in result:
                    if(j[0] in data.havedFid):
                        pts_j=getPts(j)
                        index1,_=isSim(pts_i,pts_j)
                        intersectFids.append(i[0])
                        intersectPts.append(pts_i[index1])
                        break
    start_index=None
    rows_copy=copy.deepcopy(data.havedRows)
    index=0
    for i in rows_copy:
        if(not i[0] in intersectFids):
            start_index=index
            break
        index+=1
    if(start_index==None):
        return []
    startPts=getPts(rows_copy[start_index])
    delete_index=[rows_copy[start_index][0]]
    del rows_copy[start_index]
    comp_pts=[]
    for i in [startPts[0],startPts[-1]]:
        if(not i in intersectPts):
            comp_pts.append(i)
    while(len(comp_pts)!=0):
        new_pts=[]
        for i in rows_copy:
            pts_t=getPts(i)
            for j in comp_pts:
                condition1=abs(pts_t[0][0]-j[0])<1 and abs(pts_t[0][1]-j[1])<1
                condition2=abs(pts_t[-1][0]-j[0])<1 and abs(pts_t[-1][1]-j[1])<1
                if(condition1):
                    delete_index.append(i[0])
                    if(not pts_t[-1] in intersectPts):
                        new_pts.append(pts_t[-1])
                elif(condition2):
                    delete_index.append(i[0])
                    if(not pts_t[0] in intersectPts):
                        new_pts.append(pts_t[0])
                else:
                    pass
        comp_pts=new_pts
        rows_copy=filter(lambda x: x[0] not in delete_index,rows_copy)
    return delete_index
def isSim(pts_i,pts_j):
    start=pts_i[0];end=pts_i[-1]
    if (abs(start[0] - pts_j[0][0]) < 2 and abs(start[1] - pts_j[0][1]) < 2):
        return 0,0
    if (abs(start[0] - pts_j[-1][0]) < 2 and abs(start[1] - pts_j[-1][1]) < 2):
        return 0,-1
    if (abs(end[0] - pts_j[0][0]) < 2 and abs(end[1] - pts_j[0][1]) < 2):
        return -1,0
    if (abs(end[0] - pts_j[-1][0]) < 2 and abs(end[1] - pts_j[-1][1]) < 2):
        return -1,-1
    return None,None
def computeIndexAngle(pts,i,j):
    if (pts[i][0] == pts[j][0]):
        angle = 90
    else:
        angle = np.arctan((pts[i][1] - pts[j][1]) / float(pts[i][0] - pts[j][0])) * 180 / np.pi
        if (angle < 0):
            angle += 180
    return angle
def getIndex(num):
    if(num==0):
        return 1
    if(num==-1):
        return -2
def getMatchLines(connectFidIndex,mm,result_tmp2,data):
    a=[]
    for i in mm:
        if(i[2][0]==connectFidIndex):
            continue
        pts_i=getPts(i[2])
        flag=0
        for j in result_tmp2:
            i_tmp, j_tmp = isSim(pts_i, getPts(j))
            if(i_tmp!=None):
                flag=1
                a.append([j_tmp,-1-j_tmp,j])
                break
        if(flag==1):
            break
    data.havedFid.append(a[0][2][0])
    data.havedRows.append(a[0][2])
    result_tmp2=filter(lambda x:x[0]!=a[0][2][0],result_tmp2)
    while(1):
        flag=0
        for j in result_tmp2:
            a_tmp, j_tmp = isSim(getPts(a[-1][2]), getPts(j))
            if(a_tmp!=None):
                flag=1
                data.havedFid.append(j[0])
                data.havedRows.append(j)
                a.append([j_tmp,-1-j_tmp,j])
                result_tmp2 = filter(lambda x: x[0] != j[0], result_tmp2)
        if(flag==0):
            break
    return a[-1]
def getpart(mm,info_data,data,TwoWayInfo):
    if(len(mm)==2):
        pts1=getPts(mm[0][2]);pts2=getPts(mm[1][2])
        StartOrEnd1, StartOrEnd2 = isSim(pts1, pts2)
        if(StartOrEnd1==0 or StartOrEnd1==-1):
            indexs1=0;indexs2=0
            for j in info_data:
                pts_j = getPts(j)
                if (not j[0] in data.havedFid):
                    pts1_StartOrEnd, ptsj1_StartOrEnd = isSim(pts1, pts_j)
                    pts2_StartOrEnd, ptsj2_StartOrEnd = isSim(pts2, pts_j)
                    if(pts1_StartOrEnd==None and pts2_StartOrEnd==None):
                        continue
                    elif(pts1_StartOrEnd !=None):
                        isHaveSamePlane1 = len(list(set([mm[0][2][2], mm[0][2][3], j[2], j[3]]))) == 4
                        angleDiff1 = abs(computeIndexAngle(pts1,pts1_StartOrEnd,getIndex(pts1_StartOrEnd)) - computeIndexAngle(pts_j,ptsj1_StartOrEnd,getIndex(ptsj1_StartOrEnd)))

                        if(pts1_StartOrEnd!=StartOrEnd1 and (isHaveSamePlane1 or angleDiff1<20 or angleDiff1>160)):
                            indexs1+=1
                    else:
                        isHaveSamePlane2 = len(list(set([mm[1][2][2], mm[1][2][3], j[2], j[3]]))) == 4
                        angleDiff2 = abs(
                            computeIndexAngle(pts2, pts2_StartOrEnd, getIndex(pts2_StartOrEnd)) - computeIndexAngle(
                                pts_j, ptsj2_StartOrEnd, getIndex(ptsj2_StartOrEnd)))
                        if(pts2_StartOrEnd!=StartOrEnd2 and (isHaveSamePlane2 or angleDiff2<20 or angleDiff2>160)):
                            indexs2+=1
            if(indexs1==0 or indexs2==0):
                return True
    info=[]
    if (data.isStart == True):
        for i in mm:
            info_i=[]
            startConnectLines = [];endConnectLines = []
            pts_i = getPts(i[2])
            for j in info_data:
                if(not j[0] in data.havedFid):
                    pts_j=getPts(j)
                    i_StartOrEnd,j_StartOrEnd=isSim(pts_i,pts_j)
                    if(i_StartOrEnd!=None):
                        isHaveSamePlane = len(list(set([i[2][2], i[2][3], j[2], j[3]]))) == 4
                        angle_=abs(computeIndexAngle(pts_j,j_StartOrEnd,getIndex(j_StartOrEnd))-computeAngle(pts_j))<15
                        angleDiff=abs(computeIndexAngle(pts_i,i_StartOrEnd,getIndex(i_StartOrEnd))-computeIndexAngle(pts_j,j_StartOrEnd,getIndex(j_StartOrEnd)))
                        if(i_StartOrEnd==0 and angle_ and (isHaveSamePlane or angleDiff<20 or angleDiff>160)):
                            startConnectLines.append([j_StartOrEnd,-1-j_StartOrEnd,j])
                        if(i_StartOrEnd==-1 and angle_ and (isHaveSamePlane or angleDiff<20 or angleDiff>160)):
                            endConnectLines.append([j_StartOrEnd,-1-j_StartOrEnd,j])

            if (len(startConnectLines) == 1):
                info_i.append(startConnectLines[0])
            elif (len(startConnectLines) > 1):
                angleStart = computeIndexAngle(pts_i, 0, 1);
                angleErr = []
                for m in startConnectLines:
                    pts_tmp = getPts(m[2])
                    if (m[0] == 0):
                        angle_tmp = computeIndexAngle(pts_tmp, 0, 1)
                    else:
                        angle_tmp = computeIndexAngle(pts_tmp, -1, -2)
                    angleErr.append(abs(angle_tmp - angleStart))
                angleErr_minIndex = angleErr.index(min(angleErr))
                info_i.append(startConnectLines[angleErr_minIndex])
            else:
                pass
            if (len(endConnectLines) == 1):
                info_i.append(endConnectLines[0])
            elif (len(endConnectLines) > 1):
                angleEnd = computeIndexAngle(pts_i, -1, -2);
                angleErr = []
                for m in endConnectLines:
                    pts_tmp = getPts(m[2])
                    if (m[0] == 0):
                        angle_tmp = computeIndexAngle(pts_tmp, 0, 1)
                    else:
                        angle_tmp = computeIndexAngle(pts_tmp, -1, -2)
                    angleErr.append(abs(angle_tmp - angleEnd))
                angleErr_minIndex = angleErr.index(min(angleErr))
                info_i.append(endConnectLines[angleErr_minIndex])
            else:
                pass
            info.append([i,info_i])
        data.isStart = False
    else:
        for i in mm:
            info_i=[]
            startConnectLines = [];endConnectLines = []
            pts_i = getPts(i[2])
            for j in info_data:
                if (not j[0] in data.havedFid):
                    pts_j = getPts(j)
                    i_StartOrEnd, j_StartOrEnd = isSim(pts_i, pts_j)
                    if (i_StartOrEnd != None and i_StartOrEnd==i[1]):
                        isHaveSamePlane = len(list(set([i[2][2], i[2][3], j[2], j[3]]))) == 4
                        angleDiff = abs(computeIndexAngle(pts_i, i_StartOrEnd, getIndex(i_StartOrEnd)) - computeIndexAngle(pts_j,j_StartOrEnd,getIndex(j_StartOrEnd)))
                        if (i_StartOrEnd == 0 and (isHaveSamePlane or angleDiff < 20 or angleDiff > 160)):
                            startConnectLines.append([j_StartOrEnd, -1 - j_StartOrEnd, j])
                        if (i_StartOrEnd == -1 and (isHaveSamePlane or angleDiff < 20 or angleDiff > 160)):
                            endConnectLines.append([j_StartOrEnd, -1 - j_StartOrEnd, j])

            if (len(startConnectLines) == 1):
                info_i.append(startConnectLines[0])
            elif (len(startConnectLines) > 1):
                angleStart = computeIndexAngle(pts_i, 0, 1);
                angleErr = []
                for m in startConnectLines:
                    pts_tmp = getPts(m[2])
                    if (m[0] == 0):
                        angle_tmp = computeIndexAngle(pts_tmp, 0, 1)
                    else:
                        angle_tmp = computeIndexAngle(pts_tmp, -1, -2)
                    angleErr.append(abs(angle_tmp - angleStart))
                angleErr_minIndex = angleErr.index(min(angleErr))
                info_i.append(startConnectLines[angleErr_minIndex])
            else:
                pass
            if (len(endConnectLines) == 1):
                info_i.append(endConnectLines[0])
            elif (len(endConnectLines) > 1):
                angleEnd = computeIndexAngle(pts_i, -1, -2);angleErr = []
                for m in endConnectLines:
                    pts_tmp = getPts(m[2])
                    if (m[0] == 0):
                        angle_tmp = computeIndexAngle(pts_tmp, 0, 1)
                    else:
                        angle_tmp = computeIndexAngle(pts_tmp, -1, -2)
                    angleErr.append(abs(angle_tmp - angleEnd))
                angleErr_minIndex = angleErr.index(min(angleErr))
                info_i.append(endConnectLines[angleErr_minIndex])
            else:
                pass
            info.append([i, info_i])
    final_iter=[]
    region_rows_indexs=[]
    region_rows_indexs.extend(data.havedFid)
    for i in info:
        for j in i[1]:
            region_rows_indexs.append(j[2][0])
    qf=QuickFind(info)
    for i in info:
        try:
            _, tmp_index, _ = TwoWayInfo[i[0][2]]
        except:
            continue
        for j in info:
            if(i[0][2][0]==j[0][2][0]):
                continue
            if(j[0][2][0] in tmp_index ):
                qf.union(i,j)
    infoClassifyResult=qf.getClassifyInfo()
    for single in infoClassifyResult:
        if(len(single)!=2):
            continue
        for j in range(len(single)):
            info_ii=single[j][1]
            for k in info_ii:
                try:
                    _, result_index, _ = TwoWayInfo[k[2]]
                except:
                    continue
                flag=0
                for q in result_index:
                    if (q in region_rows_indexs):
                        flag=1
                        data.havedFid.append(k[2][0])
                        data.havedRows.append(k[2])
                        final_iter.append(k)
                        break
                if(flag==0 and single[1-j][1]!=[]):
                    final_iter.append(single[j][0])
    mm_index=set([p[2][0] for p in mm])
    final_iter_index=set([q[2][0] for q in final_iter])
    if(len(final_iter)==0  or mm_index==final_iter_index):
        return True
    else:
        return getpart(final_iter,info_data,data,TwoWayInfo)

def getLinesDist(havedRows):
    Rows_x=[];Rows_y=[]
    for i in havedRows:
        pts_i=getPts(i)
        Rows_x.extend([pts_i[0][0],pts_i[-1][0]])
        Rows_y.extend([pts_i[0][1],pts_i[-1][1]])
    return max(Rows_y)-min(Rows_y),max(Rows_x)-min(Rows_x)
def deleteAllLinesOptimized(info1,info2,TwoWayInfo,shpPath):
    dist_lines=[]
    index=0
    nums=len(info1)
    numsInfo=len(info2)
    while(nums):
            try:
                i=info1[0]
            except:
                for k in range(len(info1)):
                    print info1[k]
            data = Data()
            result = []
            try:
                result_tmp, _, result_intersect = TwoWayInfo[i]
            except:
                del info1[0]
                index += 1
                nums = len(info1)
                DisplayPogressBar(index,numsInfo)
                continue
            if(len(result_tmp)==1):
                tmp_sums=[]
                try:
                    tmp, _, tmp_intersect = TwoWayInfo[result_tmp[0]]
                except:
                    del info1[0]
                    index += 1
                    nums = len(info1)
                    DisplayPogressBar(index, numsInfo)
                    continue
                tmp_sums.extend(tmp)
                tmp_sums.extend([i, result_tmp[0]])
                for j in tmp_sums:
                        result_index = [k[0] for k in result]
                        if (not j[0] in result_index):
                            result.append(j)
            else:
                result.append(i)
                result.extend(result_tmp)
            isResultGood,processedResult=processRows(result)
            if(not isResultGood):
                del info1[0]
                index += 1
                nums = len(info1)
                DisplayPogressBar(index,numsInfo)
                continue
            data.havedRows.extend(processedResult)
            data.havedFid.extend([kk[0] for kk in data.havedRows])
            input = [[-2, -2, k] for k in processedResult]
            try:
                getpart(input, info2, data,TwoWayInfo)
            except:
                del info1[0]
                index += 1
                nums = len(info1)
                DisplayPogressBar(index,numsInfo)
                continue
            dist_lines_fids=[set(k[1]) for k in dist_lines]
            havedRows=[]
            for j in data.havedRows:
                havedRowsIndex=[k[0] for k in havedRows]
                if(not j[0] in havedRowsIndex):
                    havedRows.append(j)
            data.havedRows=havedRows
            data.havedFid=set([k[0] for k in data.havedRows])
            if(not data.havedFid in dist_lines_fids):
                delete_pts_index = deleteRows_Test(data,TwoWayInfo)
                if(delete_pts_index==[]):
                    del info1[0]
                    index += 1
                    nums = len(info1)
                    DisplayPogressBar(index, numsInfo)
                    continue
                maxDist=getLinesDist(data.havedRows)
                dist_lines.append([data.havedRows,data.havedFid,maxDist,delete_pts_index])
                if(maxDist>1000):
                    tmp = list(data.havedFid)
                    if(i[0] in tmp):
                        info1=filter(lambda x:not x[0] in tmp,info1)
                    else:
                        tmp.append(i[0])
                        info1 = filter(lambda x: not x[0] in tmp, info1)
                else:
                    del info1[0]
            else:
                del info1[0]
            index+=1
            nums=len(info1)
            DisplayPogressBar(index, numsInfo)
    result_dist_lines=[]
    result_dist_lines_fids=[]
    dist_lines=sorted(dist_lines,key=lambda x:x[2],reverse=True)
    for i in dist_lines:
        if(len(result_dist_lines)==0):
            result_dist_lines.append(i)
            result_dist_lines_fids.extend(i[1])
            result_dist_lines_fids=list(set(result_dist_lines_fids))
        else:
            flag =0
            for j in i[1]:
                if(j in result_dist_lines_fids):
                    flag=1
            if(flag==0):
                result_dist_lines.append(i)
                result_dist_lines_fids.extend(i[1])
                result_dist_lines_fids = list(set(result_dist_lines_fids))
    delete_all_lines=[]
    result_dist_lines_delete=[k[3] for k in result_dist_lines]
    for i in result_dist_lines_delete:
        delete_all_lines.extend(i)

    cursor = arcpy.da.UpdateCursor(shpPath, ["OID@"])
    for i in cursor:
        if(i[0] in delete_all_lines):
            cursor.deleteRow()
    print "\ndone"
if __name__ == '__main__':
    arcpy.env.workspace = sys.argv[1]
    inputRoadName=sys.argv[2]
    blockShpName=sys.argv[3]
    outputShpName=sys.argv[4]
    arcpy.FeatureToPolygon_management(inputRoadName, blockShpName)
    arcpy.PolygonToLine_management(blockShpName, "line.shp")
    arcpy.Dissolve_management("line.shp", outputShpName, ["LEFT_FID", "RIGHT_FID"], "", "SINGLE_PART", "")
    print "step 1"
    info1, info2, TwoWayInfo=addField(outputShpName)
    print "step 2"
    deleteAllLinesOptimized(info1,info2,TwoWayInfo,outputShpName)
