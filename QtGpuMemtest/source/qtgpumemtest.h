#ifndef QTGPUMEMTEST_H
#define QTGPUMEMTEST_H

#include <QtGui>
#include <QVector>
#include <QMap>
#include <QFile>
#include <QIODevice>
#include <QSettings>

#include "../../GpuDisplayWidget/GpuDisplayWidget/gpudisplaywidget.h"
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
	void about();
	void exit();
	void copyResults();
	void exportResults();
	void clearDevices();
	void relistDevices();
	void checkAllDevices(int checked);
	void switchView();
	void startChecked();
	void stopAll();
	void returnHome() { setView(BasicView); };

	// test controller
	/*void startTests(int deviceIdx, QVector<TestInfo>& instanceTests, bool infinite = false);
	void endTests(int deviceIdx);*/

	// test controller, temporary solution by catching signals from the advanced views
	void testsStarted(int n);
	void testEnded(const int index, QString testName);
	void stressTestEnded();
	void stressTestProgress();

	void handleBlockingError(int deviceIdx, int err, int cudaErr, QString line, QString file);
	void handleNonBlockingError(int deviceIdx, int warn, QString line, QString file);
	void handleProgress(int deviceIdx, int testNo, int action);

	// Aggregate test options
	void quickTest();
	void stressTest();

private:
	Ui::QtGpuMemtestClass ui;

	ViewMode					currentViewMode;
	QList<GpuDisplayWidget*>	deviceWidgets;
	QList<cudaDeviceProp*>		devices;
	QMap<int, QtGpuThread*>		testThreads;
	QVector<TestInfo>			tests;				// default tests (default template)

	bool allTestsDone;
	QTimer* stressTimer;
	QTimer* stressSubTimer;
	int stressTestsRunning;
	bool stressTesting;
};

#endif // QTGPUMEMTEST_H
