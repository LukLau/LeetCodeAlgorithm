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
        int i = low;
        int j = mid + 1;
        int k = 0;
        while (i <= mid && j <= high) {
            if (nums[i] <= nums[j]) {
                tmp[k++] = nums[i++];
            } else {
                tmp[k++] = nums[j++];
            }
        }
        while (i <= mid) {
            tmp[k++] = nums[i++];
        }
        while (j <= high) {
            tmp[k++] = nums[j++];
        }
        if (tmp.length >= 0) {
            System.arraycopy(tmp, 0, nums, low, tmp.length);
        }
    }

    public static void main(String[] args) {
        MergeSort mergeSort = new MergeSort();

        int[] nums = new int[]{-1, 4, -1, 343, 4, -2};
        mergeSort.mergeSort(nums, 0, nums.length - 1);

        for (int num : nums) {
            System.out.println(num);
        }
    }


}
