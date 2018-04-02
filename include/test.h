#ifndef TEST_H
#define TEST_H


class test
{
    public:
        test(test1);
        virtual ~test();

        unsigned int GetCounter() { return m_Counter; }
        void SetCounter(unsigned int val) { m_Counter = val; }

    protected:

    private:
        unsigned int m_Counter;
};

#endif // TEST_H
