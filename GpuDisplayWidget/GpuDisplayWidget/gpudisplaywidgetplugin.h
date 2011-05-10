#ifndef GPUDISPLAYWIDGETPLUGIN_H
#define GPUDISPLAYWIDGETPLUGIN_H

#include <QDesignerCustomWidgetInterface>

class GpuDisplayWidgetPlugin : public QObject, public QDesignerCustomWidgetInterface
{
	Q_OBJECT
	Q_INTERFACES(QDesignerCustomWidgetInterface)

public:
	GpuDisplayWidgetPlugin(QObject *parent = 0);

	bool isContainer() const;
	bool isInitialized() const;
	QIcon icon() const;
	QString domXml() const;
	QString group() const;
	QString includeFile() const;
	QString name() const;
	QString toolTip() const;
	QString whatsThis() const;
	QWidget *createWidget(QWidget *parent);
	void initialize(QDesignerFormEditorInterface *core);

private:
	bool initialized;
};

#endif // GPUDISPLAYWIDGETPLUGIN_H
