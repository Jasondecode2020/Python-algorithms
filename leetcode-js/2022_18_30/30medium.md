### 1 986. Interval List Intersections

```js
// compare the elements from left to right, compare each interval, grab the intersection and move on
var intervalIntersection = function(firstList, secondList) {
    const res = [];
    let i = 0, j = 0;
    while (i < firstList.length && j < secondList.length) {
        let [aStart, aEnd] = firstList[i];
        let [bStart, bEnd] = secondList[j];
        if (aEnd < bStart) {
            i++;
        } else if (bEnd < aStart) {
            j++;
        } else if (aEnd <= bEnd) {
            res.push([Math.max(aStart, bStart), aEnd]);
            i++;
        } else if (bEnd <= aEnd) {
            res.push([Math.max(aStart, bStart), bEnd]);
            j++;
        }
    }
    return res;
};
```

### 2 920 · Meeting Rooms (leetcode 252 and lintcode)

Description
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

Input: intervals = [(0,30),(5,10),(15,20)]
Output: false
Explanation: 
(0,30), (5,10) and (0,30),(15,20) will conflict
```python
class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: if a person could attend all meetings
    """
    def canAttendMeetings(self, intervals):
        # Write your code here
        intervals.sort(key=lambda x:x.start)
        for i in range(1, len(intervals)):
            if intervals[i].start < intervals[i-1].end:
                return False
        return True
```

### 3 919 · Meeting Rooms II

Description
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

Input: intervals = [(0,30),(5,10),(15,20)]
Output: 2
Explanation:
We need two meeting rooms
room1: (0,30)
room2: (5,10),(15,20)

```python
class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def minMeetingRooms(self, intervals):
        # Write your code here
        start = sorted([i.start for i in intervals])
        end = sorted([i.end for i in intervals])

        res, count = 0, 0
        s, e = 0, 0
        while s < len(intervals):
            if start[s] < end[e]:
                s += 1
                count += 1
            else:
                e += 1
                count -= 1
            res = max(res, count)
        return res
```