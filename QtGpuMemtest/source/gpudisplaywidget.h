#ifndef GPUDISPLAYWIDGET_H
#define GPUDISPLAYWIDGET_H

#include <QtGui>

#include "common.h"
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

	enum Mode { SelectingMode, StoppedMode, RunningMode };

	QFont font() const { return labelGpu->font(); }
	void setFont(QFont const &);

	int index() const { return widgetIndex; }
	void setIndex(int idx) { widgetIndex = idx; }

	QString getName() { return labelGpu->text(); }

	void setGpuName(const QString& gpuName) { labelGpu->setText(gpuName); }
	void setGpuMemory(const QString& gpuMemory) { labelMemory->setText(gpuMemory); }
	void setTests(QVector<TestInfo>& aTests);
	QVector<TestInfo> getTests();

public slots:
	void setCheckStart(const int checked) { checkStart->setChecked(checked > 0); }
	void setState(Mode newState);
	
	void testFailed(TestInfo test);
	void testPassed(TestInfo test);
	void testStarting(TestInfo test);

	void maskSelect(TestInfo pivot, bool mask);

	void startChecked() { if(checkStart->checkState() == Qt::Checked) emit startTests(0); }
	void stopButtonClicked();

signals:
	void startTests(int infinite);
	void stopTests();
	void displayResults();

protected:
	virtual void paintEvent(QPaintEvent* event);

private:

	QVBoxLayout *layout;
	QHBoxLayout *innerTopLayout;
	QHBoxLayout *innerBottomLayout;

	QLabel		*labelGpu;
	QLabel		*labelMemory;
	QLabel		*labelStopping;
	QHBoxLayout	*progress;
	QCheckBox	*checkStart;

	QToolButton *startButton;
	QToolButton *startLoopButton;
	QToolButton *stopButton;
	QToolButton *resultsButton;

	int			widgetIndex;

	QVector<TestIconWidget*>	testWidgets;

	Mode		state;

};

#endif // GPUDISPLAYWIDGET_H
