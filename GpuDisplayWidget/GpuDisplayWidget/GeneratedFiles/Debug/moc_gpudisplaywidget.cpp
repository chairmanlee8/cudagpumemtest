/****************************************************************************
** Meta object code from reading C++ file 'gpudisplaywidget.h'
**
** Created: Fri Mar 11 15:23:45 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../gpudisplaywidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gpudisplaywidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GpuDisplayWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       1,   24, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      26,   18,   17,   17, 0x0a,
      60,   46,   17,   17, 0x0a,

 // properties: name, type, flags
      95,   89, 0x40095103,

       0        // eod
};

static const char qt_meta_stringdata_GpuDisplayWidget[] = {
    "GpuDisplayWidget\0\0gpuName\0setGpuName(QString)\0"
    "gpuTestStatus\0setTestStatus(GpuTestStatus)\0"
    "QFont\0font\0"
};

const QMetaObject GpuDisplayWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_GpuDisplayWidget,
      qt_meta_data_GpuDisplayWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GpuDisplayWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GpuDisplayWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GpuDisplayWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GpuDisplayWidget))
        return static_cast<void*>(const_cast< GpuDisplayWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int GpuDisplayWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: setGpuName((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: setTestStatus((*reinterpret_cast< const GpuTestStatus(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 2;
    }
#ifndef QT_NO_PROPERTIES
      else if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QFont*>(_v) = font(); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setFont(*reinterpret_cast< QFont*>(_v)); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 1;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}
QT_END_MOC_NAMESPACE
