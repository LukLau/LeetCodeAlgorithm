package org.code.algorithm.sort;

/**
 * @author dora
 * @date 2020/8/11
 */
public class HeapSort {

    public void headSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        for (int i = nums.length / 2 - 1; i >= 0; i--) {
            headAdjust(nums, i, nums.length);
        }
        for (int i = nums.length - 1; i > 0; i--) {
            swap(nums, 0, i);
            headAdjust(nums, 0, i);
        }
    }

    private void headAdjust(int[] nums, int i, int m) {
        int tmp = nums[i];
        for (int k = 2 * i + 1; k < m; k = 2 * k + 1) {
            if (k + 1 < m && nums[k] < nums[k + 1]) {
                k = k + 1;
            }
            if (nums[i] > nums[k]) {
                break;
            }
            nums[i] = nums[k];
            i = k;
        }
        nums[i] = tmp;
    }


    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public static void main(String[] args) {
        HeapSort heapSort = new HeapSort();
        int[] nums = new int[]{-1, 343, -1, 3, 4, 32};
        heapSort.headSort(nums);

        for (int num : nums) {
            System.out.println(num);
        }
    }


}
