package org.code.algorithm.datastructe;

/**
 * 278. First Bad Version
 *
 * @author luk
 * @date 2020/11/17
 */
public class VersionControl {

    public int firstBadVersion(int n) {
        if (n <= 1) {
            return n;
        }
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (!isBadVersion(mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }


    private boolean isBadVersion(int version) {
        return true;
    }
}
