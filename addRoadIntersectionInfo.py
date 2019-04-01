import numpy as np
import sys
import arcpy
import os
def DisplayPogressBar(index,numsRows):
    sys.stdout.write("\r" + "[%s%%]" % ((index + 1) * 100 / numsRows))
    sys.stdout.flush()

def getIndex(num):
    if(num==0):
        return 1
    if(num==-1):
        return -2

def getPts(row):
    pts=[]
    for part in row[1]:
        for point in part:
            pts.append([point.X,point.Y])
    return pts

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

def getTjunctionIndex(data):
    angle_err_info=[]
    angle_err = 1000;angle_err_min_index = -1;angle_max=0;angle_err_max_index=-1
    for i in range(len(data)):
        j = i + 1
        if (j == 3):
            j = 0
        angle_err_tmp = getAngle(data[i],data[j])
        angle_err_info.append(angle_err_tmp)
        if (angle_err > angle_err_tmp):
            angle_err = angle_err_tmp
            angle_err_min_index = i
        if (angle_max <= angle_err_tmp):
            angle_max = angle_err_tmp
            angle_err_max_index = i
    if(angle_err_info[angle_err_min_index]>45):
        if (angle_err_max_index == 0):
            return True, data[2], [data[1], data[0]]
        elif (angle_err_max_index == 1):
            return True, data[0], [data[1], data[2]]
        else:
            return True, data[1], [data[0], data[2]]
    else:
        if(angle_err_min_index==0):
            return False, data[2], [data[1], data[0]]
        elif(angle_err_min_index==1):
            return False, data[0], [data[1], data[2]]
        else:
            return False, data[1], [data[0], data[2]]

def isSimForRoad(pts_i,pts_j):
    start=pts_i[0];end=pts_i[-1]
    condition1=abs(start[0] - pts_j[0][0]) < 0.1 and abs(start[1] - pts_j[0][1]) < 0.1
    condition2=abs(start[0] - pts_j[-1][0]) < 0.1 and abs(start[1] - pts_j[-1][1]) < 0.1
    condition3=abs(end[0] - pts_j[0][0]) < 0.1 and abs(end[1] - pts_j[0][1]) < 0.1
    condition4=abs(end[0] - pts_j[-1][0]) < 0.1 and abs(end[1] - pts_j[-1][1]) < 0.1
    if(condition1):
        if(condition4):
            return [0,0],[-1,-1]
        else:
            return [0,0],[None,None]
    if(condition2):
        if(condition3):
            return [0,-1],[-1,0]
        else:
            return [0,-1],[None,None]
    if(condition3):
        if(condition2):
            return [-1,0],[0,-1]
        else:
            return [-1,0],[None,None]
    if(condition4):
        if(condition1):
            return [-1,-1],[0,0]
        else:
            return [-1,-1],[None,None]
    return [None,None],[None,None]

def addRoadInfo(workPath,shpPath,outputShpName):
    arcpy.env.workspace=workPath
    planeShpName="".join([shpPath.split(".")[0],"_plane",".shp"])
    lineShpName="".join([shpPath.split(".")[0],"_line",".shp"])

    arcpy.ExtendLine_edit(shpPath, "328 Feet", "EXTENSION")
    # cursor = arcpy.da.SearchCursor(shpPath, ["OID@", "SHAPE@"])
    # rows=[]
    # for i in cursor:
    #     rows.append(i)
    # deleteIndex=[]
    # for j in rows:
    #     pts_j=getPts(j)
    #     dist=np.linalg.norm(np.array(pts_j[0])-np.array(pts_j[-1]))
    #     if(dist<20):
    #         deleteIndex.append(j[0])
    #
    # cursor_update = arcpy.da.UpdateCursor(shpPath, ["OID@", "SHAPE@"])
    # for k in cursor_update:
    #     if(k[0] in deleteIndex):
    #         cursor_update.deleteRow()

    arcpy.FeatureToPolygon_management(shpPath, planeShpName)
    arcpy.PolygonToLine_management(planeShpName, lineShpName)
    arcpy.Dissolve_management(lineShpName, outputShpName, ["LEFT_FID", "RIGHT_FID"], "", "SINGLE_PART", "")

    arcpy.AddField_management(outputShpName, "Tjunction0", "TEXT")
    arcpy.AddField_management(outputShpName, "Tjunction1", "TEXT")
    arcpy.AddField_management(outputShpName, "Fork0", "TEXT")
    arcpy.AddField_management(outputShpName, "Fork1", "TEXT")
    arcpy.AddField_management(outputShpName, "CroRoad0", "TEXT")
    arcpy.AddField_management(outputShpName, "CroRoad1", "TEXT")
    arcpy.UpdateCursor(outputShpName)

    cursor_search = arcpy.da.UpdateCursor(outputShpName,
                                          ["OID@", "Tjunction0", "Tjunction1", "Tjunction1", "Fork0", "Fork1",
                                           "CroRoad0", "CroRoad1"])
    for i in cursor_search:
            i[1] = -1
            i[2] = -1
            i[3] = -1
            i[4] = -1
            i[5] = -1
            i[6] = -1
            i[7] = -1
            cursor_search.updateRow(i)
    cursor_search = arcpy.da.SearchCursor(outputShpName, ["OID@", "SHAPE@"])
    rows = []
    for row in cursor_search:
        rows.append(row)
    delete_info = []
    allInfo = {}
    for p in range(len(rows)):
        allInfo[p] = {}
    for i in range(len(rows)):
        pts_i = getPts(rows[i])
        j_index_start = [];j_index_end = []
        i_index_start = [0, rows[i]]
        i_index_end = [-1, rows[i]]
        flag0 = 0;flag1 = 0
        if ([rows[i][0], 0] in delete_info):
            flag0 = 1
        if ([rows[i][0], -1] in delete_info):
            flag1 = 1
        if (flag0 == 1 and flag1 == 1):
            continue
        for j in rows:
            pts_j = getPts(j)
            if (rows[i][0] == j[0]):
                continue
            [i_index1, j_index1], [i_index2, j_index2] = isSimForRoad(pts_i, pts_j)
            if(flag0==0):
                if (i_index1 == 0):
                        j_index_start.append([j_index1, j])
                if(i_index2==0):
                    j_index_start.append([j_index2, j])
            if(flag1==0):
                if (i_index1 == -1):
                    j_index_end.append([j_index1, j])
                if(i_index2==-1):
                    j_index_end.append([j_index2, j])
        if (len(j_index_start) == 2):
            isTjunction, TjunctionMain, TjunctionOthers = getTjunctionIndex(
                [i_index_start, j_index_start[0], j_index_start[1]])
            strings = str(TjunctionOthers[0][1][0]) + "_" + str(abs(TjunctionOthers[0][0])) + "_" + \
                      str(TjunctionOthers[1][1][0]) + "_" + str(abs(TjunctionOthers[1][0]))
            if (isTjunction):
                if (TjunctionMain[0] == 0):
                    allInfo[TjunctionMain[1][0]]["Tjunction0"] = strings
                else:
                    allInfo[TjunctionMain[1][0]]["Tjunction1"] = strings
                delete_info.extend(
                    [[TjunctionMain[1][0], TjunctionMain[0]], [TjunctionOthers[0][1][0], TjunctionOthers[0][0]],
                     [TjunctionOthers[1][1][0], TjunctionOthers[1][0]]])
            else:
                if (TjunctionMain[0] == 0):
                    allInfo[TjunctionMain[1][0]]["Fork0"] = strings
                else:
                    allInfo[TjunctionMain[1][0]]["Fork1"] = strings
                delete_info.extend(
                    [[TjunctionMain[1][0], TjunctionMain[0]], [TjunctionOthers[0][1][0], TjunctionOthers[0][0]],
                     [TjunctionOthers[1][1][0], TjunctionOthers[1][0]]])
        elif (len(j_index_start) == 3):
            strings = str(j_index_start[0][1][0]) + "_" + str(abs(j_index_start[0][0])) + "_" + \
                      str(j_index_start[1][1][0]) + "_" + str(abs(j_index_start[1][0])) + "_" + \
                      str(j_index_start[2][1][0]) + "_" + str(abs(j_index_start[2][0]))
            allInfo[i_index_start[1][0]]["CroRoad0"] = strings
            delete_info.extend(
                [[i_index_start[1][0], i_index_start[0]], [j_index_start[0][1][0], j_index_start[0][0]],
                 [j_index_start[1][1][0], j_index_start[1][0]], [j_index_start[2][1][0], j_index_start[2][0]]])
        else:
            pass
        if (len(j_index_end) == 2):
            isTjunction, TjunctionMain, TjunctionOthers = getTjunctionIndex(
                [i_index_end, j_index_end[0], j_index_end[1]])
            strings = str(TjunctionOthers[0][1][0]) + "_" + str(abs(TjunctionOthers[0][0])) + "_" + \
                      str(TjunctionOthers[1][1][0]) + "_" + str(abs(TjunctionOthers[1][0]))
            if (isTjunction):
                if (TjunctionMain[0] == 0):
                    allInfo[TjunctionMain[1][0]]["Tjunction0"] = strings
                else:
                    allInfo[TjunctionMain[1][0]]["Tjunction1"] = strings
                delete_info.extend(
                    [[TjunctionMain[1][0], TjunctionMain[0]], [TjunctionOthers[0][1][0], TjunctionOthers[0][0]],
                     [TjunctionOthers[1][1][0], TjunctionOthers[1][0]]])
            else:
                if (TjunctionMain[0] == 0):
                    allInfo[TjunctionMain[1][0]]["Fork0"] = strings
                else:
                    allInfo[TjunctionMain[1][0]]["Fork1"] = strings
                delete_info.extend(
                    [[TjunctionMain[1][0], TjunctionMain[0]], [TjunctionOthers[0][1][0], TjunctionOthers[0][0]],
                     [TjunctionOthers[1][1][0], TjunctionOthers[1][0]]])
        elif (len(j_index_end) == 3):
            strings = str(j_index_end[0][1][0]) + "_" + str(abs(j_index_end[0][0])) + "_" + \
                      str(j_index_end[1][1][0]) + "_" + str(abs(j_index_end[1][0])) + "_" + \
                      str(j_index_end[2][1][0]) + "_" + str(abs(j_index_end[2][0]))
            allInfo[i_index_end[1][0]]["CroRoad1"] = strings
            delete_info.extend(
                [[i_index_end[1][0], i_index_end[0]], [j_index_end[0][1][0], j_index_end[0][0]],
                 [j_index_end[1][1][0], j_index_end[1][0]], [j_index_end[2][1][0], j_index_end[2][0]]])
        else:
            pass
    workspace = arcpy.env.workspace
    editor = arcpy.da.Editor(workspace)
    editor.startEditing(False, True)
    editor.startOperation()
    cursor_update = arcpy.da.UpdateCursor(outputShpName,
                                          ["OID@", "SHAPE@", "Tjunction0", "Tjunction1", "Fork0", "Fork1", "CroRoad0",
                                           "CroRoad1"])
    index = 0
    for row in cursor_update:
        a = allInfo[index]
        for i, j in a.items():
            if (i == "Tjunction0"):
                row[2] = j
            elif (i == "Tjunction1"):
                row[3] = j
            elif (i == "Fork0"):
                row[4] = j
            elif (i == "Fork1"):
                row[5] = j
            elif (i == "CroRoad0"):
                row[6] = j
            elif (i == "CroRoad1"):
                row[7] = j
            else:
                pass
        index += 1
        cursor_update.updateRow(row)
    editor.stopOperation()
    editor.stopEditing(True)
if __name__ == '__main__':
    workPath=sys.argv[1]
    shpName=sys.argv[2]
    outputShpName=sys.argv[3]
    addRoadInfo(workPath,shpName,outputShpName)
    print "done"