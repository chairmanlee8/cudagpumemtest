#ifndef QTGPUMEMTEST_H
#define QTGPUMEMTEST_H

#include <QtGui>
#include <QVector>
#include <QMap>
#include <QFile>
#include <QIODevice>
#include <QSettings>

#include "gpudisplaywidget.h"
#include "ui_qtgpumemtest.h"

#include <cuda_runtime_api.h>

#include "gputests.h"

enum ViewMode { BasicView, AdvancedView, BasicResultsView, AdvancedResultsView, NoDevicesView };
class QtGpuMemtest : public QMainWindow
{
	Q_OBJECT

public:
	QtGpuMemtest(QWidget *parent = 0, Qt::WFlags flags = 0);
	~QtGpuMemtest();

public slots:
	void customStressValue(int minutes);
	void setView(ViewMode viewMode);

	// some menu and toolbar functions
	void copyResults();
	void exportResults();
	void exit();
	void about();
	void clearDevices();
	void relistDevices();
	void switchView();
	void checkAllDevices(int checked);
	void startChecked();
	void stopAll();
	void returnHome() { setView(BasicView); };

	// test controller
	/*void startTests(int deviceIdx, QVector<TestInfo>& instanceTests, bool infinite = false);
	void endTests(int deviceIdx);*/

	// test controller, temporary solution by catching signals from the advanced views
	/*void testsStarted(int n);
	void testEnded(const int index, QString testName);*/
	/*void stressTestEnded();
	void stressTestProgress();*/

	// Aggregate test options
	/*void quickTest();
	void stressTest();*/

	void widgetStartTests(int infinite);
	void widgetStopTests();
	void widgetTestsEnded();

	//void widgetDisplayResults();

private:
	Ui::QtGpuMemtestClass ui;

	ViewMode					currentViewMode;
	QList<GpuDisplayWidget*>	deviceWidgets;
	QList<cudaDeviceProp*>		devices;
	QMap<int, QtGpuThread*>		testThreads;
	QVector<TestInfo>			tests;				// default tests (default template)

	/*bool allTestsDone;
	QTimer* stressTimer;
	QTimer* stressSubTimer;
	int stressTestsRunning;
	bool stressTesting;*/
};

#endif // QTGPUMEMTEST_H
