package org.code.algorithm.sort;

/**
 * @author dora
 * @date 2020/8/11
 */
public class MergeSort {
    public void mergeSort(int[] nums, int low, int high) {
        if (low >= high) {
            return;
        }
        int mid = low + (high - low) / 2;
        mergeSort(nums, low, mid);
        mergeSort(nums, mid + 1, high);
        merge(nums, low, mid, high);
    }

    private void merge(int[] nums, int low, int mid, int high) {
        int[] tmp = new int[high - low + 1];
        int leftSide = low;
        int rightSide = mid + 1;
        int index = 0;
        while (leftSide <= mid && rightSide <= high) {
            if (nums[leftSide] <= nums[rightSide]) {
                tmp[index++] = nums[leftSide++];
            } else {
                tmp[index++] = nums[rightSide++];
            }
        }
        while (leftSide <= mid) {
            tmp[index++] = nums[leftSide++];
        }
        while (rightSide <= high) {
            tmp[index++] = nums[rightSide++];
        }
        if (tmp.length >= 0) {
            System.arraycopy(tmp, 0, nums, low, tmp.length);
        }
    }


    public static void main(String[] args) {
        MergeSort sort = new MergeSort();
        int[] nums = new int[]{-1, -2, -1, 3, 43, 2, 4, 54545};

        sort.mergeSort(nums, 0, nums.length - 1);

        for (int num : nums) {
            System.out.println(num);
        }
    }


}
