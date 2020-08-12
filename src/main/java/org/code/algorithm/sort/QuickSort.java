package org.code.algorithm.sort;

/**
 * @author dora
 * @date 2020/8/11
 */
public class QuickSort {
    public void quickSort(int[] nums, int low, int high) {
        if (low > high) {
            return;
        }
        int partition = partition(nums, low, high);

        quickSort(nums, low, partition - 1);

        quickSort(nums, partition + 1, high);

    }

    private int partition(int[] nums, int low, int high) {

        int pivot = nums[low];
        while (low < high) {
            while (low < high && pivot <= nums[high]) {
                high--;
            }
            if (low < high) {
                nums[low] = nums[high];
                low++;
            }
            while (low < high && pivot >= nums[low]) {
                low++;
            }
            if (low < high) {
                nums[high] = nums[low];
                high--;
            }
        }
        nums[low] = pivot;
        return low;
    }

    public static void main(String[] args) {
        QuickSort sort = new QuickSort();
        int[] nums = new int[]{-1, -1, 434, 32, 23, 43, 12, 4358, 98};

        sort.quickSort(nums, 0, nums.length - 1);
        for (int num : nums) {
            System.out.println(num);
        }
    }

}
