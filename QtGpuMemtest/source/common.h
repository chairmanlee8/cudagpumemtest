#ifndef __COMMON_H
#define __COMMON_H

struct MemoryError
{
	unsigned long	addr;
	unsigned long	expected;
	unsigned long	current;
	unsigned long	second_read;
};

struct TestInputParams
{
	char*			ptr;
	unsigned int	tot_num_blocks;
	int				num_iterations;
};

struct TestOutputParams
{
	MemoryError*	err_vector;
	int*			err_count;
};

typedef int (*TestFunc)(TestInputParams*, TestOutputParams*, bool*);

enum TestStatus { TestNotStarted, TestPassed, TestFailed, TestRunning };

#ifndef __CUDACC__

#include <QString>

struct TestInfo
{
	int			testId;
	QString		testName;
	QString		testShortName;		// short name, no more than 2 characters long, for identifying the test in the TestIconWidget.
	TestFunc	testFunc;
	bool		testEnabled;

	TestInfo() : testId(-1), testName(QString("")), testShortName(QString("")), testFunc(0), testEnabled(false) { };
	TestInfo(int id, QString name, QString shortName, TestFunc func, bool e) : testId(id), testName(name), testShortName(shortName), testFunc(func), testEnabled(e) {};

	bool TestInfo::operator==(const TestInfo& other) { return (testId == other.testId); };
	bool TestInfo::operator!=(const TestInfo& other) { return !(*this == other); };
};

#endif

#endif