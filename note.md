### Get afs on local 

```sshfs -o reconnect login@ssh.cri.epita.fr:/afs/cri.epita.fr/user/x/xa/xavier.login/u/ afs```

warning
The files will not be accessible after the Kerberos ticket expires (usually after a few days). If this happens, ask for a new ticket by running `kinit -f login@CRI.EPITA.FR` and unmount the afs folder by running `umount afs/`. You can then mount it again with the same sshfs command as previously.


### Access CRI GPU

```afs ssh -X -l clement.gregoire -p 22001 gpgpu.image.lrde.iaas.epita.fr```


### Doc
```
https://docs.forge.epita.fr/epita/environment/from-home/afs/
https://moodle.epita.fr/mod/page/view.php?id=84549
```
