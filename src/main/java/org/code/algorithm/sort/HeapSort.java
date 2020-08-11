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
            adjustHeap(nums, i, nums.length);
        }
        for (int i = nums.length - 1; i > 0; i--) {

            swap(nums, 0, i);

            adjustHeap(nums, 0, i);
        }
    }

    private void adjustHeap(int[] nums, int i, int length) {
        int tmp = nums[i];
        for (int k = 2 * i + 1; k < length; k = 2 * k + 1) {
            if (k + 1 < length && nums[k] < nums[k + 1]) {
                k = k + 1;
            }
            if (nums[k] > tmp) {
                nums[i] = nums[k];
                i = k;
            } else {
                break;
            }
        }
        nums[i] = tmp;
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public static void main(String[] args) {
        HeapSort sort = new HeapSort();
        int[] nums = new int[]{-1, -1, 3, 2, 43, 32, 1};
        sort.headSort(nums);
        for (int num : nums) {
            System.out.println(num);
        }
    }


}
