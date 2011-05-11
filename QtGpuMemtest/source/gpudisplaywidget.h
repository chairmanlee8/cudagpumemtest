#ifndef GPUDISPLAYWIDGET_H
#define GPUDISPLAYWIDGET_H

#include <QtGui>

#include "testiconwidget.h"
#include "resultsdisplay.h"
#include "gputests.h"

class GpuDisplayWidget : public QWidget
{
	Q_OBJECT

	Q_PROPERTY(QFont font READ font WRITE setFont)

public:
	GpuDisplayWidget(QWidget *parent = 0);
	~GpuDisplayWidget();

	QFont font() const;
	void setFont(QFont const &);

	int index() const;
	void setIndex(int idx);

	QtGpuMemtest* controller() { return m_controller; };
	void setController(QtGpuMemtest* a) { m_controller = a; };

	QString getLog();
	QString getName() { return m_labelGpu->text(); };

public slots:
	void setGpuName(const QString& gpuName);
	void setGpuMemory(const QString& gpuMemory);
	void setTestStatus(const TestStatus& gpuTestStatus);
	void setCheckStart(const int checked);
	void setTests(QVector<TestInfo>& aTests);

	void startTest(bool infinite = false);
	void startTestOnce();	// TODO: must be a better solution than this crap
	void startTestInfinite();
	void endTest();
	void displayLog();

	void testFailed(int deviceIdx, QString testName);
	void testPassed(int deviceIdx, QString testName);
	void testStarting(int deviceIdx, QString testName);
	void testLog(int deviceIdx, QString testName, QString logMessage);

	bool isChecked() { return m_checkStart->checkState() == Qt::Checked; };
	bool isTestFailed();

signals:
	void testStarted(const int index, bool infinite);
	void testEnded(const int index);

protected:
	virtual void paintEvent(QPaintEvent* event);

private:
	QtGpuMemtest *m_controller;

	QVBoxLayout *layout;
	QHBoxLayout *innerTopLayout;
	QHBoxLayout *innerBottomLayout;

	QLabel		*m_labelGpu;
	QLabel		*m_labelMemory;
	QLabel		*m_labelStopping;
	QHBoxLayout	*m_progress;
	QCheckBox	*m_checkStart;

	QToolButton *startButton;
	QToolButton *startLoopButton;
	QToolButton *startStressButton;
	QToolButton *stopButton;
	QToolButton *resultsButton;

	int			m_widgetIndex;

	QVector<TestInfo>			tests;
	QVector<TestIconWidget*>	testWidgets;
	QVector<QString>			log;

	QtGpuThread	*gpuThread;

};

#endif // GPUDISPLAYWIDGET_H
