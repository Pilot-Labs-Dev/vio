//
// Created by root on 5/12/21.
//

#ifndef VIO_FOR_IT_H
#define VIO_FOR_IT_H
#include <iostream>
#include <vector>
#include <tuple>

//https://stackoverflow.com/questions/24881799/get-index-in-c11-foreach-loop
// Wrapper class
template <typename T>
class for_impl
{
public:
    // The return value of the operator* of the iterator, this
    // is what you will get inside of the for loop
    struct item
    {
        size_t index;
        typename T::value_type & item;
    };
    typedef item value_type;

    // Custom iterator with minimal interface
    struct iterator
    {
        iterator(typename T::iterator _it, size_t counter=0) :
                it(_it), counter(counter)
        {}

        iterator operator++()
        {
            return iterator(++it, ++counter);
        }

        bool operator!=(iterator other)
        {
            return it != other.it;
        }

        typename T::iterator::value_type item()
        {
            return *it;
        }

        value_type operator*()
        {
            return value_type{counter, *it};
        }

        size_t index()
        {
            return counter;
        }

    private:
        typename T::iterator it;
        size_t counter;
    };

    for_impl(T & t) : container(t) {}

    iterator begin()
    {
        return iterator(container.begin());
    }

    iterator end()
    {
        return iterator(container.end());
    }

private:
    T & container;
};


// A templated free function allows you to create the wrapper class
// conveniently
template <typename T>
for_impl<T> _for(T & t)
{
    return for_impl<T>(t);
}
#endif //VIO_FOR_IT_H
