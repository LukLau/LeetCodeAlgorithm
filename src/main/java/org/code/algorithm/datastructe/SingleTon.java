package org.code.algorithm.datastructe;

/**
 * @author luk
 * @date 2020/11/16
 */
public class SingleTon {
    private static volatile SingleTon instance = null;


    private SingleTon() {
    }

    public static SingleTon getInstance() {
        if (instance == null) {
            synchronized (SingleTon.class) {
                if (instance == null) {
                    instance = new SingleTon();
                }
            }
        }
        return instance;
    }
}
