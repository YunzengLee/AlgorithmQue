package com.imooc;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class MyLruCache<K,V>{

    public static void main(String[] args) {
        //非并发测试
        MyLruCache<Integer, String> myLruCache = new MyLruCache<>(3);
        myLruCache.put(1,"java");
        System.out.println(myLruCache.get(1));
        myLruCache.put(2,"python");
        myLruCache.put(3,"c++");
        System.out.println(myLruCache.get(2));
        myLruCache.put(4,"C");
        myLruCache.put(5,"PHP");
        System.out.println(myLruCache.get(2));
        
    }


    private final int maxCapacity;
    private ConcurrentHashMap<K,V> cacheMap;
    private ConcurrentLinkedQueue<K> keys;

    private ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
    private Lock write = readWriteLock.writeLock();
    private Lock read = readWriteLock.readLock();

    public MyLruCache(int maxCapacity){
        if (maxCapacity <0 ){
            throw new IllegalArgumentException("Illegal max capacity:"+maxCapacity);
        }
        this.maxCapacity = maxCapacity;
        cacheMap = new ConcurrentHashMap<>(maxCapacity);
        keys = new ConcurrentLinkedQueue<>();
    }

    public V put(K key, V value){
        //加写锁
        write.lock();
        try{
            //key是否存在当前缓存
            if(cacheMap.containsKey(key)){
                moveToTailOfQueue(key);
                cacheMap.put(key,value);
                return value;
            }
            //是否超出缓存容量
            if(cacheMap.size() == maxCapacity){
                System.out.println("maxCapacity of cache reached");
                removeOldestKey();
            }
            //key不存在于当前缓存,将key添加到队尾，并缓存
            keys.add(key);
            cacheMap.put(key,value);
            return value;

        }finally {
            write.unlock();
        }
    }

    public V get(K key){
        //加读锁
        read.lock();
        try{
            //key是否存在于缓存中
            if(cacheMap.containsKey(key)){
                moveToTailOfQueue(key);
                return cacheMap.get(key);
            }
            //不存在就返回null
            return null;

        }finally {
            read.unlock();
        }
    }

    public V remove(K key){
        write.lock();
        try{
            //key是否存在于当前缓存中
            if(cacheMap.containsKey(key)){
                keys.remove(key);
                return cacheMap.remove(key);
            }
            //不存在缓存中就返回null
            return null;
        }finally {
            write.unlock();
        }
    }

    /**
     * 移到尾部
     * @param key
     */
    private void moveToTailOfQueue(K key){
        keys.remove(key);
        keys.add(key);
    }

    private void removeOldestKey(){
        K oldestKey = keys.poll();
        if (oldestKey != null){
            cacheMap.remove(oldestKey);
        }
    }

    public int size(){
        return cacheMap.size();
    }
}
